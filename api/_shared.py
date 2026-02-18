"""Shared utilities for Vercel serverless API endpoints.

NOT a serverless function itself (underscore prefix).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# CORS & Response helpers
# ---------------------------------------------------------------------------

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
}


def json_response(handler: BaseHTTPRequestHandler, data: dict | list, status: int = 200) -> None:
    """Send a JSON response with CORS headers."""
    body = json.dumps(data, default=_json_serializer).encode()
    handler.send_response(status)
    for key, val in CORS_HEADERS.items():
        handler.send_header(key, val)
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 500) -> None:
    """Send a JSON error response."""
    json_response(handler, {"error": message}, status)


def options_response(handler: BaseHTTPRequestHandler) -> None:
    """Handle CORS preflight."""
    handler.send_response(200)
    for key, val in CORS_HEADERS.items():
        handler.send_header(key, val)
    handler.end_headers()


def _json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def get_query_params(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    """Parse query string from the request path."""
    parsed = urlparse(handler.path)
    params = parse_qs(parsed.query)
    return {k: v[0] for k, v in params.items()}


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests dependency for lightweight endpoints)
# ---------------------------------------------------------------------------

def http_get(url: str, headers: dict | None = None, timeout: int = 15) -> dict | list:
    """Simple GET request returning parsed JSON."""
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post(url: str, data: dict, headers: dict | None = None, timeout: int = 15) -> dict:
    """Simple POST request returning parsed JSON."""
    body = json.dumps(data).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=body, headers=hdrs, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Coinbase Advanced Trade API
# ---------------------------------------------------------------------------

COINBASE_API_BASE = "https://api.coinbase.com"


def coinbase_request(
    method: str,
    path: str,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> dict | list:
    """Make an authenticated request to the Coinbase Advanced Trade API.

    Uses HMAC-SHA256 signing: sign(timestamp + method + path + body).
    """
    api_key = api_key or os.environ.get("COINBASE_API_KEY", "")
    api_secret = api_secret or os.environ.get("COINBASE_SECRET", "")

    if not api_key or not api_secret:
        raise ValueError("COINBASE_API_KEY and COINBASE_SECRET are required")

    timestamp = str(int(time.time()))
    message = timestamp + method.upper() + path + ""
    # Coinbase legacy API keys have base64-encoded secrets
    decoded_secret = base64.b64decode(api_secret)
    signature = base64.b64encode(
        hmac.new(
            decoded_secret,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }

    url = COINBASE_API_BASE + path
    return http_get(url, headers=headers)


# ---------------------------------------------------------------------------
# CoinGecko API
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def coingecko_get(path: str, params: dict | None = None) -> dict | list:
    """GET request to CoinGecko public API."""
    url = COINGECKO_BASE + path
    if params:
        url += "?" + urlencode(params)
    api_key = os.environ.get("COINGECKO_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    return http_get(url, headers=headers)


# ---------------------------------------------------------------------------
# Binance Public API
# ---------------------------------------------------------------------------

BINANCE_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


def binance_ohlcv(symbol: str, interval: str = "4h", limit: int = 200) -> list[dict]:
    """Fetch OHLCV candles from Binance public API.

    Args:
        symbol: Trading pair like "BTCUSDT"
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1000)

    Returns:
        List of dicts with keys: timestamp, open, high, low, close, volume
    """
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = http_get(url)
    return [
        {
            "timestamp": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }
        for candle in raw
    ]


def binance_ticker_24h(symbol: str) -> dict:
    """Fetch 24h ticker stats from Binance."""
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr?symbol={symbol}"
    return http_get(url)


def binance_funding_rates(limit: int = 20) -> list[dict]:
    """Fetch current funding rates from Binance Futures."""
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
    data = http_get(url)
    # Sort by absolute funding rate (most interesting first) and limit
    data.sort(key=lambda x: abs(float(x.get("lastFundingRate", 0))), reverse=True)
    return data[:limit]


# ---------------------------------------------------------------------------
# Crypto metadata (symbol → CoinGecko ID, name, color)
# ---------------------------------------------------------------------------

CRYPTO_META = {
    "BTC": {"id": "bitcoin", "name": "Bitcoin", "color": "#f7931a"},
    "ETH": {"id": "ethereum", "name": "Ethereum", "color": "#627eea"},
    "SOL": {"id": "solana", "name": "Solana", "color": "#9945ff"},
    "USDC": {"id": "usd-coin", "name": "USD Coin", "color": "#2775ca"},
    "USDT": {"id": "tether", "name": "Tether", "color": "#26a17b"},
    "LINK": {"id": "chainlink", "name": "Chainlink", "color": "#2a5ada"},
    "AVAX": {"id": "avalanche-2", "name": "Avalanche", "color": "#e84142"},
    "DOGE": {"id": "dogecoin", "name": "Dogecoin", "color": "#c2a633"},
    "ADA": {"id": "cardano", "name": "Cardano", "color": "#0033ad"},
    "DOT": {"id": "polkadot", "name": "Polkadot", "color": "#e6007a"},
    "MATIC": {"id": "matic-network", "name": "Polygon", "color": "#8247e5"},
    "UNI": {"id": "uniswap", "name": "Uniswap", "color": "#ff007a"},
    "ATOM": {"id": "cosmos", "name": "Cosmos", "color": "#2e3148"},
    "XRP": {"id": "ripple", "name": "XRP", "color": "#23292f"},
    "LTC": {"id": "litecoin", "name": "Litecoin", "color": "#bfbbbb"},
    "SHIB": {"id": "shiba-inu", "name": "Shiba Inu", "color": "#ffa409"},
    "ARB": {"id": "arbitrum", "name": "Arbitrum", "color": "#28a0f0"},
    "OP": {"id": "optimism", "name": "Optimism", "color": "#ff0420"},
    "NEAR": {"id": "near", "name": "NEAR Protocol", "color": "#00c08b"},
    "FIL": {"id": "filecoin", "name": "Filecoin", "color": "#0090ff"},
    "APT": {"id": "aptos", "name": "Aptos", "color": "#4ed8a0"},
    "SUI": {"id": "sui", "name": "Sui", "color": "#6fbcf0"},
    "PEPE": {"id": "pepe", "name": "Pepe", "color": "#4a8c34"},
    "RENDER": {"id": "render-token", "name": "Render", "color": "#000000"},
    "INJ": {"id": "injective-protocol", "name": "Injective", "color": "#00f2fe"},
    "TIA": {"id": "celestia", "name": "Celestia", "color": "#7b2bf9"},
    "SEI": {"id": "sei-network", "name": "Sei", "color": "#9b1c1c"},
    "AAVE": {"id": "aave", "name": "Aave", "color": "#b6509e"},
    "CRV": {"id": "curve-dao-token", "name": "Curve", "color": "#ff0000"},
    "MKR": {"id": "maker", "name": "Maker", "color": "#1aab9b"},
}

# Stock metadata
STOCK_META = {
    "AAPL": {"name": "Apple Inc.", "color": "#a2aaad"},
    "NVDA": {"name": "NVIDIA Corp", "color": "#76b900"},
    "TSLA": {"name": "Tesla Inc.", "color": "#cc0000"},
    "MSFT": {"name": "Microsoft Corp", "color": "#00a4ef"},
    "GOOGL": {"name": "Alphabet Inc.", "color": "#4285f4"},
    "AMZN": {"name": "Amazon.com Inc.", "color": "#ff9900"},
    "META": {"name": "Meta Platforms", "color": "#0668e1"},
}

STABLECOINS = {"USDC", "USDT", "BUSD", "DAI", "UST", "TUSD", "USDP", "FRAX", "USD"}


def get_asset_meta(symbol: str) -> dict:
    """Get metadata for a symbol (name, color, coingecko id)."""
    sym = symbol.upper()
    if sym in CRYPTO_META:
        return CRYPTO_META[sym]
    if sym in STOCK_META:
        return STOCK_META[sym]
    return {"name": sym, "color": "#888888"}
