"""GET /api/opportunities — Real actionable opportunities from portfolio + analysis."""

from __future__ import annotations

import logging
import os
from http.server import BaseHTTPRequestHandler

from api._shared import (
    STABLECOINS,
    coinbase_request,
    coingecko_get,
    error_response,
    get_asset_meta,
    http_get,
    json_response,
    options_response,
    CRYPTO_META,
)

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            opps = _build_opportunities()
            json_response(self, opps)
        except Exception as e:
            logger.exception("Opportunities endpoint failed")
            error_response(self, str(e))

    def do_OPTIONS(self):
        options_response(self)


def _build_opportunities() -> list[dict]:
    """Generate real opportunities by combining portfolio, signals, and sentiment."""

    # 1. Get portfolio holdings
    holdings = _get_holdings_with_values()
    if not holdings:
        return []

    total_value = sum(h["value"] for h in holdings)
    cash = sum(h["value"] for h in holdings if h["symbol"] in STABLECOINS or h["symbol"] == "USD")

    # 2. Get signals (with technical data)
    signals_by_symbol = _get_signals_data(holdings)

    # 3. Get market regime
    regime = _get_regime()

    # 4. Generate opportunities
    opportunities = []
    opp_id = 1

    for h in holdings:
        symbol = h["symbol"]
        value = h["value"]
        alloc_pct = (value / total_value * 100) if total_value > 0 else 0

        if symbol in STABLECOINS or symbol == "USD":
            # Cash opportunities
            opps = _cash_opportunities(opp_id, h, total_value, regime)
            opportunities.extend(opps)
            opp_id += len(opps)
            continue

        sig = signals_by_symbol.get(symbol, {})
        if not sig:
            continue

        rsi = sig.get("_rsi", 50)
        trend_bullish = sig.get("_trend_bullish", False)
        trend_bearish = sig.get("_trend_bearish", False)
        macd_hist = sig.get("_macd_hist", 0)
        price = sig.get("price", 0)
        atr = sig.get("_atr", 0)

        # --- TAKE PROFIT (overbought + large position) ---
        if rsi > 70 and alloc_pct > 5:
            sell_pct = 25 if rsi > 80 else 15
            opportunities.append({
                "id": opp_id,
                "priority": 1 if rsi > 80 else 2,
                "type": "take_profit",
                "symbol": symbol,
                "headline": f"Take {sell_pct}% profit on {symbol} - RSI overbought at {rsi:.0f}",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"{symbol} is showing overbought conditions with RSI at {rsi:.0f}. "
                    f"Chart indicators suggest momentum may be fading. "
                    f"Consider taking {sell_pct}% off the table to lock in gains."
                ),
                "steps": [
                    f"Sell {sell_pct}% of your {symbol} position (~${value * sell_pct / 100:,.0f})",
                    "Convert to USDC or preferred stablecoin",
                    f"Set a trailing stop on remainder" + (f" at ${price * 0.92:,.2f}" if price else ""),
                    "Reassess on RSI pullback below 50",
                ],
                "potentialGain": f"Lock in ~${value * sell_pct / 100:,.0f}",
                "risk": f"Price continues higher and you miss {sell_pct}% of further gains",
                "confidence": min(85, int(50 + rsi / 4)),
            })
            opp_id += 1

        # --- BUY THE DIP (oversold) ---
        if rsi < 30 and not trend_bearish:
            add_amount = total_value * 0.03
            opportunities.append({
                "id": opp_id,
                "priority": 2,
                "type": "buy_the_dip",
                "symbol": symbol,
                "headline": f"Add to {symbol} - RSI oversold at {rsi:.0f}",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"{symbol} is oversold (RSI {rsi:.0f}) without being in a confirmed downtrend. "
                    f"This is often a buying opportunity for a mean reversion bounce."
                ),
                "steps": [
                    f"Buy ~${add_amount:,.0f} more {symbol}",
                    f"Set stop loss at ${price * 0.92:,.2f}" if price else "Set an 8% stop loss",
                    "Scale in: 50% now, 50% if it dips another 3-5%",
                ],
                "potentialGain": "Mean reversion bounce of 5-15%",
                "risk": "Oversold can become more oversold in a crash",
                "confidence": 55,
            })
            opp_id += 1

        # --- CUT LOSS (bearish trend) ---
        if trend_bearish and alloc_pct > 5:
            opportunities.append({
                "id": opp_id,
                "priority": 1,
                "type": "cut_loss",
                "symbol": symbol,
                "headline": f"Cut loss on {symbol} - bearish trend confirmed",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"{symbol} is in a confirmed downtrend (price below 50 and 200 SMA, "
                    f"MACD bearish). Reducing exposure prevents further drawdown."
                ),
                "steps": [
                    f"Sell 50% of your {symbol} position (~${value * 0.5:,.0f})",
                    "Move proceeds to USDC",
                    "Set alert at prior support to re-enter if trend reverses",
                ],
                "potentialGain": f"Avoid further losses on ${value:,.0f} position",
                "risk": "Price reverses immediately after selling (whipsaw)",
                "confidence": 60,
            })
            opp_id += 1

        # --- RIDE THE TREND (bullish, healthy RSI) ---
        if trend_bullish and 40 < rsi < 65:
            opportunities.append({
                "id": opp_id,
                "priority": 3,
                "type": "ride_the_trend",
                "symbol": symbol,
                "headline": f"Hold {symbol} - bullish trend with healthy momentum",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"{symbol} is in a confirmed uptrend with RSI at {rsi:.0f} showing "
                    f"momentum without being overbought. Keep riding the trend."
                ),
                "steps": [
                    f"Hold your {symbol} position (${value:,.0f})",
                    f"Set trailing stop at ${price * 0.90:,.2f}" if price else "Set a 10% trailing stop",
                    "Consider adding on pullbacks to 50 SMA",
                ],
                "potentialGain": "Trend continuation toward next resistance",
                "risk": "Sudden reversal or macro event",
                "confidence": 60,
            })
            opp_id += 1

        # --- ROTATE TO USDC (greed market + overbought) ---
        if regime in ("greed", "euphoria") and rsi > 65 and alloc_pct > 15:
            rotate_pct = 30 if regime == "euphoria" else 20
            opportunities.append({
                "id": opp_id,
                "priority": 1 if regime == "euphoria" else 2,
                "type": "rotate_to_usdc",
                "symbol": symbol,
                "headline": f"Rotate {rotate_pct}% of {symbol} to USDC - {regime} market + overbought",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"Market sentiment is at {regime} levels and {symbol} RSI is elevated. "
                    f"This combination historically precedes corrections. Take some off the table."
                ),
                "steps": [
                    f"Sell {rotate_pct}% of {symbol} (~${value * rotate_pct / 100:,.0f})",
                    "Convert to USDC",
                    "Set buy-back alert at -10% to -15% from current price",
                ],
                "potentialGain": f"Protect ${value * rotate_pct / 100:,.0f} from potential correction",
                "risk": "Market continues higher without you for the rotated portion",
                "confidence": 65 if regime == "euphoria" else 50,
            })
            opp_id += 1

        # --- REBALANCE (over-concentrated) ---
        if alloc_pct > 30:
            target = 25
            excess_pct = alloc_pct - target
            trim_value = total_value * excess_pct / 100
            opportunities.append({
                "id": opp_id,
                "priority": 2,
                "type": "rebalance",
                "symbol": symbol,
                "headline": f"Trim {symbol} - over-concentrated at {alloc_pct:.0f}%",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"Your {symbol} position is {alloc_pct:.1f}% of portfolio, exceeding "
                    f"the recommended {target}% limit. Trimming spreads risk."
                ),
                "steps": [
                    f"Sell ~${trim_value:,.0f} of {symbol} to bring allocation to {target}%",
                    "Spread proceeds across other assets or hold in USDC",
                ],
                "potentialGain": "Reduced concentration risk",
                "risk": "Trimmed asset outperforms in the short term",
                "confidence": 70,
            })
            opp_id += 1

        # --- HEDGE WITH FUTURES (large position + MACD bearish) ---
        if alloc_pct > 20 and macd_hist < 0 and value > 1000:
            opportunities.append({
                "id": opp_id,
                "priority": 2,
                "type": "hedge_with_futures",
                "symbol": symbol,
                "headline": f"Hedge {symbol} with 30% futures short - MACD turning bearish",
                "currentValue": round(value, 2),
                "allocationPct": round(alloc_pct, 1),
                "unrealizedPnlPct": 0,
                "description": (
                    f"Your {symbol} position is {alloc_pct:.1f}% of portfolio with MACD showing "
                    f"bearish momentum. A partial hedge protects downside while keeping spot."
                ),
                "steps": [
                    f"Open 1x short perp on {symbol} worth ~${value * 0.3:,.0f}",
                    "Use Binance, Bybit, or OKX for futures",
                    "Set stop on hedge at +5% above entry",
                    "Close hedge when MACD crosses bullish",
                ],
                "potentialGain": f"Offset up to 30% of losses if {symbol} drops 10-20%",
                "risk": "Funding costs + loss on short if price pumps",
                "confidence": 55,
            })
            opp_id += 1

    # Portfolio-level opportunities
    cash_pct = (cash / total_value * 100) if total_value > 0 else 0
    if regime in ("greed", "euphoria") and cash_pct < 15:
        target_cash = total_value * 0.20
        deficit = target_cash - cash
        opportunities.append({
            "id": opp_id,
            "priority": 2,
            "type": "rotate_to_usdc",
            "symbol": "PORTFOLIO",
            "headline": f"Raise cash to 20% - only {cash_pct:.0f}% in stables during {regime}",
            "currentValue": round(cash, 2),
            "allocationPct": round(cash_pct, 1),
            "unrealizedPnlPct": 0,
            "description": (
                f"Cash allocation is {cash_pct:.1f}% during a {regime} market. "
                f"Raising to 20% gives dry powder for the inevitable correction."
            ),
            "steps": [
                f"Trim most overbought positions by ~${deficit:,.0f} total",
                "Prioritize trimming positions with RSI > 70",
                "Move proceeds to USDC",
                "Deploy aggressively when sentiment drops below 30",
            ],
            "potentialGain": f"${deficit:,.0f} ready to buy the dip",
            "risk": "Market keeps rallying with less exposure",
            "confidence": 60,
        })

    # Sort by priority then confidence
    opportunities.sort(key=lambda o: (o["priority"], -o["confidence"]))

    return opportunities


def _get_holdings_with_values() -> list[dict]:
    """Get held assets with current USD values."""
    try:
        data = coinbase_request("GET", "/api/v3/brokerage/accounts")
    except Exception as e:
        logger.error(f"Coinbase fetch failed: {e}")
        return []

    accounts = data.get("accounts", [])
    holdings = []

    # Collect symbols for price fetch
    symbols = []
    raw_holdings = []
    for acct in accounts:
        bal = float(acct.get("available_balance", {}).get("value", 0) or 0)
        hold_val = float(acct.get("hold", {}).get("value", 0) or 0)
        total = bal + hold_val
        if total <= 0:
            continue
        currency = acct.get("currency", "").upper()
        if not currency:
            continue
        raw_holdings.append({"symbol": currency, "qty": total})
        if currency not in STABLECOINS and currency != "USD" and currency in CRYPTO_META:
            symbols.append(currency)

    # Fetch prices
    prices = {}
    if symbols:
        try:
            ids = []
            sym_to_id = {}
            for s in symbols:
                meta = CRYPTO_META.get(s)
                if meta:
                    ids.append(meta["id"])
                    sym_to_id[meta["id"]] = s
            data = coingecko_get("/simple/price", {
                "ids": ",".join(ids),
                "vs_currencies": "usd",
            })
            for cg_id, pd in data.items():
                sym = sym_to_id.get(cg_id)
                if sym:
                    prices[sym] = pd.get("usd", 0)
        except Exception:
            pass

    for h in raw_holdings:
        sym = h["symbol"]
        qty = h["qty"]
        if sym in STABLECOINS or sym == "USD":
            holdings.append({"symbol": sym, "value": qty, "qty": qty, "price": 1.0})
        elif sym in prices:
            price = prices[sym]
            holdings.append({"symbol": sym, "value": qty * price, "qty": qty, "price": price})

    return holdings


def _get_signals_data(holdings: list[dict]) -> dict:
    """Get signal data for each symbol (reuses signals endpoint logic)."""
    import pandas as pd
    import ta

    signals = {}
    for h in holdings:
        symbol = h["symbol"]
        if symbol in STABLECOINS or symbol == "USD":
            continue
        if symbol not in CRYPTO_META:
            continue

        try:
            pair = f"{symbol}USDT"
            candles = binance_ohlcv(pair, interval="4h", limit=200)
            if len(candles) < 50:
                continue

            df = pd.DataFrame(candles)
            close = df["close"]
            high = df["high"]
            low = df["low"]
            last_close = float(close.iloc[-1])

            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            macd_hist = ta.trend.MACD(close).macd_diff()
            sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
            sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator()
            atr_val = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

            last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            last_macd = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0
            last_sma50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else last_close
            last_sma200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else last_close
            last_atr = float(atr_val.iloc[-1]) if not pd.isna(atr_val.iloc[-1]) else last_close * 0.02

            signals[symbol] = {
                "price": last_close,
                "_rsi": last_rsi,
                "_macd_hist": last_macd,
                "_trend_bullish": last_close > last_sma50 > last_sma200,
                "_trend_bearish": last_close < last_sma50 < last_sma200,
                "_sma_50": last_sma50,
                "_atr": last_atr,
            }
        except Exception as e:
            logger.warning(f"Failed to get signals for {symbol}: {e}")

    return signals


def _get_regime() -> str:
    """Get current market regime from fear & greed."""
    try:
        data = http_get("https://api.alternative.me/fng/?limit=1&format=json")
        fg = int(data["data"][0]["value"])
        if fg <= 20:
            return "capitulation"
        if fg <= 35:
            return "fear"
        if fg <= 55:
            return "neutral"
        if fg <= 75:
            return "greed"
        return "euphoria"
    except Exception:
        return "neutral"


def _cash_opportunities(opp_id: int, holding: dict, total: float, regime: str) -> list[dict]:
    """Generate opportunities for cash/stablecoin positions."""
    value = holding["value"]
    if value < 100:
        return []

    opps = []

    if regime in ("fear", "capitulation") and value > 500:
        deploy_pct = 30 if regime == "capitulation" else 15
        deploy_amount = value * deploy_pct / 100
        opps.append({
            "id": opp_id,
            "priority": 2,
            "type": "buy_the_dip",
            "symbol": "PORTFOLIO",
            "headline": f"Deploy {deploy_pct}% of cash (~${deploy_amount:,.0f}) - {regime} market buying",
            "currentValue": round(value, 2),
            "allocationPct": round(value / total * 100, 1) if total > 0 else 0,
            "unrealizedPnlPct": 0,
            "description": (
                f"Market is in '{regime}' mode. Deploying cash into quality assets "
                f"at fear-discounted prices has historically been very profitable."
            ),
            "steps": [
                f"Deploy ~${deploy_amount:,.0f} across your top conviction assets",
                "Split evenly or weight toward the most oversold",
                f"Keep the remaining ${value - deploy_amount:,.0f} as reserve",
                "Set stop losses at -10% on new purchases",
            ],
            "potentialGain": "Buying fear has historically returned 20-50% within 6 months",
            "risk": "Market drops further before recovering",
            "confidence": 50,
        })

    return opps
