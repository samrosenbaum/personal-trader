"""GET /api/sentiment — Real market sentiment data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

from api._shared import (
    binance_funding_rates,
    coingecko_get,
    error_response,
    http_get,
    json_response,
    options_response,
)

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            sentiment = _build_sentiment()
            json_response(self, sentiment)
        except Exception as e:
            logger.exception("Sentiment endpoint failed")
            error_response(self, str(e))

    def do_OPTIONS(self):
        options_response(self)


def _build_sentiment() -> dict:
    """Fetch and assemble all sentiment data."""

    # 1. Fear & Greed Index
    fear_greed = _fetch_fear_greed()

    # 2. Market overview from CoinGecko
    market_overview = _fetch_market_overview()

    # 3. Funding rates from Binance
    funding = _fetch_funding_rates()

    # 4. Trending coins from CoinGecko
    trending = _fetch_trending()

    # 5. Compute direction index
    direction_index = _compute_direction_index(fear_greed, market_overview, funding)

    return {
        "fearGreedIndex": fear_greed.get("value", 50),
        "fearGreedLabel": fear_greed.get("label", "Neutral"),
        "history": fear_greed.get("history", []),
        "marketOverview": market_overview,
        "trendingCoins": trending,
        "fundingRates": funding,
        "directionIndex": direction_index,
    }


def _fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index from alternative.me."""
    try:
        data = http_get("https://api.alternative.me/fng/?limit=30&format=json")
        entries = data.get("data", [])
        if not entries:
            return {"value": 50, "label": "Neutral", "history": []}

        current = entries[0]
        value = int(current["value"])
        label = current["value_classification"]

        history = []
        for entry in entries:
            ts = int(entry["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "value": int(entry["value"]),
            })

        return {"value": value, "label": label, "history": history}
    except Exception as e:
        logger.error(f"Fear & Greed fetch failed: {e}")
        return {"value": 50, "label": "Neutral", "history": []}


def _fetch_market_overview() -> dict:
    """Fetch global market data from CoinGecko."""
    try:
        data = coingecko_get("/global")
        gd = data.get("data", {})

        total_cap = gd.get("total_market_cap", {}).get("usd", 0)
        total_vol = gd.get("total_volume", {}).get("usd", 0)

        # Format large numbers
        def fmt_large(n):
            if n >= 1e12:
                return f"{n / 1e12:.2f}T"
            if n >= 1e9:
                return f"{n / 1e9:.1f}B"
            if n >= 1e6:
                return f"{n / 1e6:.1f}M"
            return str(round(n))

        return {
            "totalMarketCap": fmt_large(total_cap),
            "volume24h": fmt_large(total_vol),
            "btcDominance": round(gd.get("market_cap_percentage", {}).get("btc", 0), 1),
            "ethDominance": round(gd.get("market_cap_percentage", {}).get("eth", 0), 1),
            "marketCapChange24h": round(gd.get("market_cap_change_percentage_24h_usd", 0), 2),
            "activeCryptos": gd.get("active_cryptocurrencies", 0),
        }
    except Exception as e:
        logger.error(f"CoinGecko global fetch failed: {e}")
        return {
            "totalMarketCap": "N/A",
            "volume24h": "N/A",
            "btcDominance": 0,
            "ethDominance": 0,
            "marketCapChange24h": 0,
            "activeCryptos": 0,
        }


def _fetch_funding_rates() -> list[dict]:
    """Fetch funding rates from Binance Futures."""
    try:
        raw = binance_funding_rates(limit=10)
        rates = []
        for item in raw:
            symbol = item.get("symbol", "")
            # Only include major pairs
            if not symbol.endswith("USDT"):
                continue
            rate = float(item.get("lastFundingRate", 0))
            rates.append({
                "symbol": symbol.replace("USDT", ""),
                "rate": round(rate * 100, 4),  # as percentage
                "annualized": round(rate * 100 * 3 * 365, 2),  # 3 funding periods/day × 365
            })
        return rates[:8]
    except Exception as e:
        logger.error(f"Binance funding rates fetch failed: {e}")
        return []


def _fetch_trending() -> list[str]:
    """Fetch trending coins from CoinGecko."""
    try:
        data = coingecko_get("/search/trending")
        coins = data.get("coins", [])
        return [c["item"]["symbol"].upper() for c in coins[:8]]
    except Exception as e:
        logger.error(f"CoinGecko trending fetch failed: {e}")
        return []


def _compute_direction_index(
    fear_greed: dict,
    market_overview: dict,
    funding: list[dict],
) -> dict:
    """Compute a market direction index from -100 (bearish) to +100 (bullish).

    Components:
    1. Fear & Greed (contrarian) — 25% weight
    2. Market cap 24h change — 25% weight
    3. Funding rate pressure — 25% weight
    4. BTC dominance trend — 25% weight
    """
    components = []

    # 1. Fear & Greed (contrarian: extreme fear = bullish, extreme greed = bearish)
    fg_val = fear_greed.get("value", 50)
    # Invert: 0 (extreme fear) → +100 (bullish), 100 (extreme greed) → -100 (bearish)
    fg_score = (50 - fg_val) * 2
    components.append({
        "name": "Fear & Greed (Contrarian)",
        "rawValue": fg_val,
        "score": round(fg_score, 1),
        "weight": 0.25,
        "description": f"Index at {fg_val} ({fear_greed.get('label', 'N/A')}). Contrarian signal: extreme fear = buy, extreme greed = sell.",
    })

    # 2. Market cap 24h change
    cap_change = market_overview.get("marketCapChange24h", 0)
    # ±5% maps to ±100
    cap_score = max(-100, min(100, cap_change * 20))
    components.append({
        "name": "Market Cap Momentum",
        "rawValue": cap_change,
        "score": round(cap_score, 1),
        "weight": 0.25,
        "description": f"Total market cap changed {cap_change:+.2f}% in 24h.",
    })

    # 3. Funding rate pressure (high positive = bearish — overleveraged longs)
    avg_funding = 0
    if funding:
        rates = [f["rate"] for f in funding if f.get("rate")]
        if rates:
            avg_funding = sum(rates) / len(rates)
    # Positive funding = bearish pressure, negative = bullish
    funding_score = max(-100, min(100, -avg_funding * 1000))
    components.append({
        "name": "Funding Rate Pressure",
        "rawValue": round(avg_funding, 4),
        "score": round(funding_score, 1),
        "weight": 0.25,
        "description": f"Avg funding rate {avg_funding:+.4f}%. High positive = overleveraged longs (bearish).",
    })

    # 4. BTC dominance trend
    btc_dom = market_overview.get("btcDominance", 50)
    # >55% = risk-off (slightly bearish for alts), <45% = alt season (bullish)
    dom_score = (50 - btc_dom) * 4  # centered at 50%, ±25% range maps to ±100
    dom_score = max(-100, min(100, dom_score))
    components.append({
        "name": "BTC Dominance",
        "rawValue": btc_dom,
        "score": round(dom_score, 1),
        "weight": 0.25,
        "description": f"BTC dominance at {btc_dom:.1f}%. >55% = risk-off, <45% = alt season.",
    })

    # Weighted sum
    total_score = sum(c["score"] * c["weight"] for c in components)
    total_score = max(-100, min(100, total_score))

    # Label
    if total_score <= -60:
        label = "STRONGLY BEARISH"
    elif total_score <= -20:
        label = "BEARISH"
    elif total_score <= 20:
        label = "NEUTRAL"
    elif total_score <= 60:
        label = "BULLISH"
    else:
        label = "STRONGLY BULLISH"

    return {
        "score": round(total_score, 1),
        "label": label,
        "components": components,
    }
