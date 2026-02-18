"""GET /api/portfolio — Real portfolio data from Coinbase."""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler

from api._shared import (
    STABLECOINS,
    coinbase_request,
    coingecko_get,
    error_response,
    get_asset_meta,
    json_response,
    options_response,
)

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            portfolio = _build_portfolio()
            json_response(self, portfolio)
        except Exception as e:
            logger.exception("Portfolio endpoint failed")
            error_response(self, str(e))

    def do_OPTIONS(self):
        options_response(self)


def _build_portfolio() -> dict:
    """Fetch Coinbase accounts, get live prices, build dashboard-compatible response."""

    # 1. Fetch Coinbase accounts
    accounts = _fetch_coinbase_accounts()

    # Check for error info from the fetch
    if not accounts or (len(accounts) == 1 and "_error" in accounts[0]):
        error_msg = accounts[0]["_error"] if accounts and "_error" in accounts[0] else "No accounts"
        return {
            "portfolioSummary": {
                "totalValue": 0,
                "cashAvailable": 0,
                "change24h": 0,
                "change24hUsd": 0,
                "marketRegime": "neutral",
            },
            "holdings": [],
            "risk": _empty_risk(),
            "_debug": error_msg,
        }

    # 2. Get live prices from CoinGecko
    symbols = [a["symbol"] for a in accounts if a["symbol"] not in STABLECOINS]
    prices = _fetch_prices(symbols)

    # 3. Build holdings with live prices and 24h changes
    holdings = []
    total_value = 0.0
    cash_available = 0.0
    total_prev_value = 0.0  # for 24h change calculation

    for acct in accounts:
        symbol = acct["symbol"]
        qty = acct["balance"]
        if qty <= 0:
            continue

        meta = get_asset_meta(symbol)

        if symbol in STABLECOINS or symbol == "USD":
            current_price = 1.0
            price_change_24h_pct = 0.0
            cash_available += qty
        elif symbol in prices:
            current_price = prices[symbol]["usd"]
            price_change_24h_pct = prices[symbol].get("usd_24h_change", 0) or 0
        else:
            # Unknown asset — skip
            continue

        value = qty * current_price
        prev_price = current_price / (1 + price_change_24h_pct / 100) if price_change_24h_pct else current_price
        prev_value = qty * prev_price

        total_value += value
        total_prev_value += prev_value

        holdings.append({
            "symbol": symbol,
            "name": meta.get("name", symbol),
            "qty": qty,
            "avgCost": 0,  # Would need trade history for real cost basis
            "currentPrice": round(current_price, 2),
            "source": "Coinbase",
            "color": meta.get("color", "#888888"),
            "_value": value,  # temporary, for allocation calc
            "_change_pct": price_change_24h_pct,
        })

    # 4. Compute allocations and clean up
    for h in holdings:
        h["value"] = round(h["_value"], 2)
        h["allocation"] = round((h["_value"] / total_value * 100) if total_value > 0 else 0, 2)
        h["costBasis"] = 0
        h["unrealizedPnl"] = 0  # Need cost basis for this
        del h["_value"]
        del h["_change_pct"]

    # Sort by value descending
    holdings.sort(key=lambda h: h["value"], reverse=True)

    # 5. Compute 24h change
    change_24h_usd = total_value - total_prev_value
    change_24h_pct = ((total_value - total_prev_value) / total_prev_value * 100) if total_prev_value > 0 else 0

    # 6. Determine market regime from fear/greed (lightweight — just fetch the score)
    market_regime = _get_market_regime()

    # 7. Build risk assessment
    risk = _compute_risk(holdings, total_value, cash_available)

    return {
        "portfolioSummary": {
            "totalValue": round(total_value, 2),
            "cashAvailable": round(cash_available, 2),
            "change24h": round(change_24h_pct, 2),
            "change24hUsd": round(change_24h_usd, 2),
            "marketRegime": market_regime,
        },
        "holdings": holdings,
        "risk": risk,
    }


def _fetch_coinbase_accounts() -> list[dict]:
    """Fetch account balances from Coinbase Advanced Trade API."""
    api_key = os.environ.get("COINBASE_API_KEY", "")
    api_secret = os.environ.get("COINBASE_SECRET", "")

    if not api_key or not api_secret:
        logger.error("Coinbase API keys not configured")
        return []

    try:
        data = coinbase_request("GET", "/api/v3/brokerage/accounts", api_key, api_secret)
    except Exception as e:
        logger.error(f"Coinbase API call failed: {e}")
        return []

    if not isinstance(data, dict):
        logger.error(f"Unexpected Coinbase response type: {type(data)}")
        return []

    accounts = data.get("accounts", [])
    result = []

    for acct in accounts:
        available = float(acct.get("available_balance", {}).get("value", 0) or 0)
        hold = float(acct.get("hold", {}).get("value", 0) or 0)
        total_balance = available + hold

        if total_balance <= 0:
            continue

        currency = acct.get("currency", "").upper()
        if not currency:
            continue

        result.append({
            "symbol": currency,
            "balance": total_balance,
            "available": available,
            "hold": hold,
            "account_name": acct.get("name", ""),
        })

    return result


def _fetch_prices(symbols: list[str]) -> dict:
    """Fetch current prices and 24h changes from CoinGecko."""
    from api._shared import CRYPTO_META

    # Map symbols to CoinGecko IDs
    ids = []
    symbol_to_id = {}
    for sym in symbols:
        meta = CRYPTO_META.get(sym.upper())
        if meta:
            ids.append(meta["id"])
            symbol_to_id[meta["id"]] = sym.upper()

    if not ids:
        return {}

    try:
        data = coingecko_get("/simple/price", {
            "ids": ",".join(ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        })
    except Exception as e:
        logger.error(f"CoinGecko price fetch failed: {e}")
        return {}

    prices = {}
    for cg_id, price_data in data.items():
        sym = symbol_to_id.get(cg_id)
        if sym:
            prices[sym] = price_data

    return prices


def _get_market_regime() -> str:
    """Fetch fear & greed index and map to regime."""
    try:
        from api._shared import http_get
        data = http_get("https://api.alternative.me/fng/?limit=1&format=json")
        fg_value = int(data["data"][0]["value"])
        if fg_value <= 20:
            return "capitulation"
        if fg_value <= 35:
            return "fear"
        if fg_value <= 55:
            return "neutral"
        if fg_value <= 75:
            return "greed"
        return "euphoria"
    except Exception:
        return "neutral"


def _compute_risk(holdings: list[dict], total_value: float, cash: float) -> dict:
    """Compute risk metrics from portfolio composition."""
    if not holdings or total_value <= 0:
        return _empty_risk()

    allocations = [h["allocation"] for h in holdings]
    top_concentration = max(allocations) if allocations else 0
    stablecoin_pct = (cash / total_value * 100) if total_value > 0 else 0

    # Diversification score: 1 = perfectly diversified, 0 = single asset
    n = len(holdings)
    if n <= 1:
        diversification = 0.0
    else:
        # Herfindahl index based
        hhi = sum((a / 100) ** 2 for a in allocations)
        diversification = round(1 - hhi, 2)

    # Warnings
    warnings = []
    suggestions = []

    if stablecoin_pct < 15:
        warnings.append(f"Stablecoin allocation at {stablecoin_pct:.1f}% - consider increasing for risk management")
    if top_concentration > 25:
        top_symbol = holdings[0]["symbol"] if holdings else "?"
        warnings.append(f"{top_symbol} concentration at {top_concentration:.1f}% of portfolio")
    if n < 5:
        warnings.append(f"Only {n} assets - consider diversifying")

    if top_concentration > 30:
        suggestions.append(f"Consider trimming {holdings[0]['symbol']} to reduce concentration risk")
    if stablecoin_pct < 10:
        suggestions.append("Consider rotating some profits into stablecoins")
    if diversification < 0.5:
        suggestions.append("Add uncorrelated assets to improve diversification")

    # Overall risk level
    risk_score = 0
    if top_concentration > 40:
        risk_score += 2
    elif top_concentration > 25:
        risk_score += 1
    if stablecoin_pct < 5:
        risk_score += 2
    elif stablecoin_pct < 15:
        risk_score += 1
    if diversification < 0.4:
        risk_score += 1

    if risk_score >= 4:
        overall = "CRITICAL"
    elif risk_score >= 3:
        overall = "HIGH"
    elif risk_score >= 1:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    return {
        "overallRisk": overall,
        "diversificationScore": diversification,
        "stablecoinPct": round(stablecoin_pct, 2),
        "topConcentrationPct": round(top_concentration, 2),
        "portfolioVolatility": 0,  # Would need historical data
        "warnings": warnings,
        "suggestions": suggestions,
    }


def _empty_risk() -> dict:
    return {
        "overallRisk": "LOW",
        "diversificationScore": 0,
        "stablecoinPct": 0,
        "topConcentrationPct": 0,
        "portfolioVolatility": 0,
        "warnings": ["No exchange connected - add API keys to see portfolio"],
        "suggestions": ["Configure COINBASE_API_KEY and COINBASE_SECRET in environment variables"],
    }
