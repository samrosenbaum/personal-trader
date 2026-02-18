"""GET /api/signals — Real trading signals from technical analysis."""

from __future__ import annotations

import logging
import os
from http.server import BaseHTTPRequestHandler

from api._shared import (
    binance_ohlcv,
    coinbase_request,
    error_response,
    get_asset_meta,
    get_query_params,
    json_response,
    options_response,
    CRYPTO_META,
    STABLECOINS,
)

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            params = get_query_params(self)
            symbols_param = params.get("symbols", "")

            if symbols_param:
                symbols = [s.strip().upper() for s in symbols_param.split(",")]
            else:
                # Default: get held symbols from Coinbase
                symbols = _get_held_symbols()

            signals = _generate_signals(symbols)
            json_response(self, signals)
        except Exception as e:
            logger.exception("Signals endpoint failed")
            error_response(self, str(e))

    def do_OPTIONS(self):
        options_response(self)


def _get_held_symbols() -> list[str]:
    """Get crypto symbols from Coinbase accounts."""
    try:
        data = coinbase_request("GET", "/api/v3/brokerage/accounts")
        accounts = data.get("accounts", [])
        symbols = []
        for acct in accounts:
            bal = float(acct.get("available_balance", {}).get("value", 0) or 0)
            hold = float(acct.get("hold", {}).get("value", 0) or 0)
            if bal + hold <= 0:
                continue
            currency = acct.get("currency", "").upper()
            if currency in STABLECOINS or currency == "USD":
                continue
            if currency in CRYPTO_META:
                symbols.append(currency)
        return symbols or ["BTC", "ETH", "SOL"]
    except Exception:
        return ["BTC", "ETH", "SOL"]


def _generate_signals(symbols: list[str]) -> list[dict]:
    """Generate trading signals for each symbol using technical analysis."""
    import pandas as pd
    import ta

    signals = []

    for symbol in symbols:
        try:
            # Fetch OHLCV from Binance
            pair = f"{symbol}USDT"
            candles = binance_ohlcv(pair, interval="4h", limit=200)
            if len(candles) < 50:
                continue

            df = pd.DataFrame(candles)
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            last_close = float(close.iloc[-1])

            # Compute indicators
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            macd_ind = ta.trend.MACD(close)
            macd_line = macd_ind.macd()
            macd_signal = macd_ind.macd_signal()
            macd_hist = macd_ind.macd_diff()
            sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
            sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator()
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

            last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            last_macd_hist = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0
            last_sma_50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else last_close
            last_sma_200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else last_close
            last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else last_close * 0.02

            # Determine trend
            trend_bullish = last_close > last_sma_50 > last_sma_200
            trend_bearish = last_close < last_sma_50 < last_sma_200

            # Generate signal
            score = 0  # -100 to +100
            reasons = []

            # RSI
            if last_rsi > 70:
                score -= 25
                reasons.append(f"RSI overbought at {last_rsi:.0f}")
            elif last_rsi < 30:
                score += 25
                reasons.append(f"RSI oversold at {last_rsi:.0f}")
            elif 40 <= last_rsi <= 60:
                reasons.append(f"RSI neutral at {last_rsi:.0f}")
            elif last_rsi > 60:
                score += 10
                reasons.append(f"RSI bullish momentum at {last_rsi:.0f}")
            else:
                score -= 10
                reasons.append(f"RSI weakening at {last_rsi:.0f}")

            # Trend
            if trend_bullish:
                score += 30
                reasons.append("Price above 50 & 200 SMA (bullish trend)")
            elif trend_bearish:
                score -= 30
                reasons.append("Price below 50 & 200 SMA (bearish trend)")
            elif last_close > last_sma_50:
                score += 15
                reasons.append("Price above 50 SMA")
            elif last_close < last_sma_50:
                score -= 15
                reasons.append("Price below 50 SMA")

            # MACD
            if last_macd_hist > 0:
                score += 15
                reasons.append("MACD histogram positive (bullish momentum)")
            else:
                score -= 15
                reasons.append("MACD histogram negative (bearish momentum)")

            # Map score to action
            if score >= 50:
                action = "strong_buy"
                urgency = "HIGH"
            elif score >= 20:
                action = "buy"
                urgency = "MEDIUM"
            elif score >= -20:
                action = "hold"
                urgency = "LOW"
            elif score >= -50:
                action = "sell"
                urgency = "MEDIUM"
            else:
                action = "strong_sell"
                urgency = "HIGH"

            # Override urgency for extreme RSI
            if last_rsi > 80 or last_rsi < 20:
                urgency = "IMMEDIATE"

            # Confidence based on signal agreement
            confidence = min(90, max(20, 50 + abs(score) // 2))

            # Price levels
            entry = round(last_close, 2)
            stop_loss = round(last_close - 2 * last_atr, 2)
            take_profit = round(last_close + 3 * last_atr, 2)
            if action in ("sell", "strong_sell"):
                # For sell signals, invert the levels
                stop_loss = round(last_close + 2 * last_atr, 2)
                take_profit = round(last_close - 3 * last_atr, 2)

            meta = get_asset_meta(symbol)
            signals.append({
                "symbol": symbol,
                "price": round(last_close, 2),
                "action": action,
                "urgency": urgency,
                "confidence": confidence,
                "entry": entry,
                "stopLoss": stop_loss,
                "takeProfit": take_profit,
                "timeframe": "4h",
                "reasons": reasons,
                # Extra data for opportunities endpoint
                "_rsi": last_rsi,
                "_macd_hist": last_macd_hist,
                "_trend_bullish": trend_bullish,
                "_trend_bearish": trend_bearish,
                "_sma_50": last_sma_50,
                "_atr": last_atr,
                "_score": score,
            })

        except Exception as e:
            logger.warning(f"Signal generation failed for {symbol}: {e}")
            continue

    # Sort by urgency then confidence
    urgency_order = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    signals.sort(key=lambda s: (urgency_order.get(s["urgency"], 4), -s["confidence"]))

    return signals
