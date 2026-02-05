"""Trading signal generator.

Combines technical indicators, chart patterns, and sentiment data
into actionable trading signals with specific recommendations:
- Move to USDC (risk-off)
- Hold current positions
- Buy / Add to position
- Sell / Reduce position
- Hedge with futures (short)
- Sell puts for income
- Buy puts for crash protection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from personal_trader.analysis.patterns import Bias, PatternReport
from personal_trader.analysis.sentiment import SentimentLevel, SentimentReport
from personal_trader.config import RiskProfile

logger = logging.getLogger(__name__)


class Action(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    MOVE_TO_STABLE = "move_to_stablecoin"
    HEDGE_FUTURES = "hedge_with_futures"
    SELL_PUTS = "sell_puts"
    BUY_PUTS = "buy_puts"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class Signal:
    symbol: str
    action: Action
    urgency: Urgency
    confidence: float  # 0-1
    reasons: list[str]
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timeframe: str = ""

    @property
    def summary(self) -> str:
        emoji_map = {
            Action.STRONG_BUY: "BUY",
            Action.BUY: "BUY",
            Action.HOLD: "HOLD",
            Action.SELL: "SELL",
            Action.STRONG_SELL: "SELL",
            Action.MOVE_TO_STABLE: "USDC",
            Action.HEDGE_FUTURES: "HEDGE",
            Action.SELL_PUTS: "PUTS",
            Action.BUY_PUTS: "PROTECT",
        }
        tag = emoji_map.get(self.action, "")
        return f"[{tag}] {self.symbol}: {self.action.value} ({self.confidence:.0%} conf, {self.urgency.value} urgency)"


@dataclass
class SignalReport:
    signals: list[Signal] = field(default_factory=list)
    market_regime: str = "unknown"
    summary: str = ""

    @property
    def top_signals(self) -> list[Signal]:
        return sorted(self.signals, key=lambda s: s.confidence, reverse=True)[:5]


class SignalGenerator:
    """Generates trading signals by combining all analysis layers."""

    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE) -> None:
        self.risk_profile = risk_profile

    def generate(
        self,
        symbol: str,
        indicators: dict[str, pd.DataFrame],
        patterns: PatternReport | None = None,
        sentiment: SentimentReport | None = None,
    ) -> list[Signal]:
        """Generate signals for a symbol.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            indicators: timeframe -> DataFrame with indicator columns
            patterns: detected chart patterns
            sentiment: current market sentiment
        """
        signals = []

        # Score from each analysis layer
        ta_scores = self._score_technical(indicators)
        pattern_score = self._score_patterns(patterns) if patterns else 0
        sentiment_score = self._score_sentiment(sentiment) if sentiment else 0

        # Weighted composite score (-100 to +100)
        weights = {"technical": 0.50, "pattern": 0.25, "sentiment": 0.25}
        composite = (
            ta_scores["composite"] * weights["technical"]
            + pattern_score * weights["pattern"]
            + sentiment_score * weights["sentiment"]
        )

        # Determine primary action
        action, urgency, confidence = self._score_to_action(composite, ta_scores, sentiment)
        reasons = self._build_reasons(ta_scores, pattern_score, sentiment_score, patterns, sentiment)

        # Calculate entry/stop/target from technical levels
        entry, stop, target = self._compute_levels(indicators, action)

        primary = Signal(
            symbol=symbol,
            action=action,
            urgency=urgency,
            confidence=confidence,
            reasons=reasons,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            timeframe="multi",
        )
        signals.append(primary)

        # Add derivative strategy signals if applicable
        deriv_signals = self._derivative_signals(symbol, composite, ta_scores, sentiment, indicators)
        signals.extend(deriv_signals)

        return signals

    def _score_technical(self, indicators: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Score technical indicators from -100 (bearish) to +100 (bullish)."""
        scores = {"trend": 0, "momentum": 0, "volatility": 0, "volume": 0, "composite": 0}
        timeframe_weights = {"15m": 0.1, "1h": 0.25, "4h": 0.35, "1d": 0.3}
        total_weight = 0

        for tf, df in indicators.items():
            if df is None or df.empty:
                continue
            w = timeframe_weights.get(tf, 0.2)
            total_weight += w
            last = df.iloc[-1]

            # --- Trend score ---
            trend = 0
            if "sma_50" in df.columns and "sma_200" in df.columns:
                if last["close"] > last["sma_50"] > last["sma_200"]:
                    trend += 30
                elif last["close"] < last["sma_50"] < last["sma_200"]:
                    trend -= 30
                elif last["close"] > last["sma_50"]:
                    trend += 15
                elif last["close"] < last["sma_50"]:
                    trend -= 15

            if "macd_hist" in df.columns:
                macd_h = last["macd_hist"]
                if macd_h > 0:
                    trend += 15
                else:
                    trend -= 15
                # MACD histogram trend (accelerating/decelerating)
                if len(df) > 2:
                    prev_h = df.iloc[-2]["macd_hist"]
                    if macd_h > prev_h:
                        trend += 5
                    else:
                        trend -= 5

            if "adx" in df.columns and last.get("adx"):
                if last["adx"] > 25:
                    if last.get("adx_pos", 0) > last.get("adx_neg", 0):
                        trend += 10
                    else:
                        trend -= 10

            if "psar" in df.columns:
                if last.get("psar_up") and not pd.isna(last["psar_up"]):
                    trend += 10
                elif last.get("psar_down") and not pd.isna(last["psar_down"]):
                    trend -= 10

            scores["trend"] += trend * w

            # --- Momentum score ---
            momentum = 0
            if "rsi_14" in df.columns and last.get("rsi_14"):
                rsi = last["rsi_14"]
                if rsi < 30:
                    momentum += 25  # oversold = bullish
                elif rsi < 40:
                    momentum += 10
                elif rsi > 70:
                    momentum -= 25  # overbought = bearish
                elif rsi > 60:
                    momentum -= 10

            if "stoch_k" in df.columns and last.get("stoch_k"):
                stoch = last["stoch_k"]
                if stoch < 20:
                    momentum += 15
                elif stoch > 80:
                    momentum -= 15

            if "mfi" in df.columns and last.get("mfi"):
                mfi = last["mfi"]
                if mfi < 20:
                    momentum += 10
                elif mfi > 80:
                    momentum -= 10

            if "cci" in df.columns and last.get("cci"):
                cci = last["cci"]
                if cci < -100:
                    momentum += 10
                elif cci > 100:
                    momentum -= 10

            scores["momentum"] += momentum * w

            # --- Volatility score ---
            vol_score = 0
            if "bb_pct" in df.columns and last.get("bb_pct") is not None:
                bb_pct = last["bb_pct"]
                if bb_pct > 1:
                    vol_score -= 15  # above upper BB = overextended
                elif bb_pct < 0:
                    vol_score += 15  # below lower BB = oversold

            if "squeeze_on" in df.columns and last.get("squeeze_on"):
                vol_score += 5  # squeeze building = energy for breakout

            scores["volatility"] += vol_score * w

            # --- Volume score ---
            vol = 0
            if "vol_ratio" in df.columns and last.get("vol_ratio"):
                vr = last["vol_ratio"]
                if vr > 2.0:
                    # High volume - direction depends on price action
                    if last["close"] > last["open"]:
                        vol += 15
                    else:
                        vol -= 15
                elif vr < 0.5:
                    vol -= 5  # low volume = weak conviction

            if "cmf" in df.columns and last.get("cmf"):
                if last["cmf"] > 0.1:
                    vol += 10
                elif last["cmf"] < -0.1:
                    vol -= 10

            scores["volume"] += vol * w

        if total_weight > 0:
            for key in scores:
                if key != "composite":
                    scores[key] = scores[key] / total_weight

        scores["composite"] = (
            scores["trend"] * 0.35
            + scores["momentum"] * 0.30
            + scores["volatility"] * 0.15
            + scores["volume"] * 0.20
        )

        return scores

    def _score_patterns(self, patterns: PatternReport) -> float:
        """Convert pattern report to a score from -100 to +100."""
        if not patterns or not patterns.patterns:
            return 0

        total = 0
        for p in patterns.patterns:
            weight = p.confidence * 40
            if p.bias == Bias.BULLISH:
                total += weight
            elif p.bias == Bias.BEARISH:
                total -= weight

        return max(-100, min(100, total))

    def _score_sentiment(self, sentiment: SentimentReport) -> float:
        """Convert sentiment report to a score.

        Note: Extreme fear is contrarian bullish, extreme greed is contrarian bearish.
        """
        if not sentiment:
            return 0

        score = 0
        if sentiment.fear_greed:
            fg = sentiment.fear_greed.value
            # Contrarian: extreme fear = buying opportunity
            if fg <= 15:
                score += 40
            elif fg <= 25:
                score += 20
            elif fg >= 85:
                score -= 40
            elif fg >= 75:
                score -= 20

        if sentiment.market_overview:
            cap_change = sentiment.market_overview.market_cap_change_24h
            if cap_change > 5:
                score += 10
            elif cap_change < -5:
                score -= 10

        if sentiment.funding_rates:
            avg_funding = np.mean([f.funding_rate for f in sentiment.funding_rates])
            if avg_funding > 0.001:
                score -= 15  # high positive funding = overleveraged longs
            elif avg_funding < -0.001:
                score += 15  # negative funding = overleveraged shorts

        return max(-100, min(100, score))

    def _score_to_action(
        self,
        composite: float,
        ta_scores: dict,
        sentiment: SentimentReport | None,
    ) -> tuple[Action, Urgency, float]:
        """Map composite score to a specific action."""
        confidence = min(abs(composite) / 100, 1.0)

        # Check for crash protection scenarios
        if composite < -60:
            if sentiment and sentiment.fear_greed and sentiment.fear_greed.value > 70:
                # Market is greedy but signals are very bearish = potential crash
                return Action.MOVE_TO_STABLE, Urgency.IMMEDIATE, confidence
            return Action.STRONG_SELL, Urgency.HIGH, confidence

        if composite < -35:
            return Action.SELL, Urgency.MEDIUM, confidence

        if composite < -15:
            return Action.SELL, Urgency.LOW, confidence

        if composite > 60:
            return Action.STRONG_BUY, Urgency.HIGH, confidence

        if composite > 35:
            return Action.BUY, Urgency.MEDIUM, confidence

        if composite > 15:
            return Action.BUY, Urgency.LOW, confidence

        return Action.HOLD, Urgency.LOW, confidence * 0.5

    def _derivative_signals(
        self,
        symbol: str,
        composite: float,
        ta_scores: dict,
        sentiment: SentimentReport | None,
        indicators: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """Generate derivative strategy signals (futures, options)."""
        signals = []

        # Hedge with futures short when moderately bearish
        if -60 < composite < -25:
            signals.append(Signal(
                symbol=symbol,
                action=Action.HEDGE_FUTURES,
                urgency=Urgency.MEDIUM,
                confidence=min(abs(composite) / 80, 0.85),
                reasons=[
                    "Bearish technical signals suggest hedging exposure",
                    "Consider shorting futures equal to a portion of your spot holdings",
                    "This protects downside while maintaining long-term spot position",
                ],
                timeframe="4h-1d",
            ))

        # Buy puts for crash protection when signals are mixed but risk is elevated
        if composite < -15 and sentiment and sentiment.fear_greed:
            fg = sentiment.fear_greed.value
            if fg > 60:
                signals.append(Signal(
                    symbol=symbol,
                    action=Action.BUY_PUTS,
                    urgency=Urgency.MEDIUM,
                    confidence=0.6,
                    reasons=[
                        f"Market greed ({fg}/100) combined with bearish technicals",
                        "Puts provide downside protection with limited upside cost",
                        "Consider 10-20% OTM puts expiring 30-60 days out",
                    ],
                    timeframe="1d-1w",
                ))

        # Sell puts for income in high-fear environments with bullish technicals
        if composite > 10 and sentiment and sentiment.fear_greed:
            fg = sentiment.fear_greed.value
            if fg < 30:
                signals.append(Signal(
                    symbol=symbol,
                    action=Action.SELL_PUTS,
                    urgency=Urgency.LOW,
                    confidence=0.55,
                    reasons=[
                        f"Fear index at {fg}/100 - high premiums available",
                        "Technical signals lean bullish - puts likely expire worthless",
                        "Sell puts at support levels to collect premium or acquire at discount",
                    ],
                    timeframe="1d-1w",
                ))

        return signals

    def _compute_levels(
        self, indicators: dict[str, pd.DataFrame], action: Action
    ) -> tuple[float | None, float | None, float | None]:
        """Compute entry, stop-loss, and take-profit prices."""
        # Use the highest-resolution available timeframe for levels
        for tf in ("1h", "4h", "1d", "15m"):
            if tf in indicators and not indicators[tf].empty:
                df = indicators[tf]
                last = df.iloc[-1]
                close = float(last["close"])
                atr = float(last.get("atr_14", close * 0.02))

                if action in (Action.BUY, Action.STRONG_BUY):
                    entry = close
                    stop = close - (atr * 2)
                    target = close + (atr * 3)
                    return round(entry, 2), round(stop, 2), round(target, 2)
                elif action in (Action.SELL, Action.STRONG_SELL, Action.MOVE_TO_STABLE):
                    entry = close
                    stop = close + (atr * 2)
                    target = close - (atr * 3)
                    return round(entry, 2), round(stop, 2), round(target, 2)
                else:
                    return close, None, None

        return None, None, None

    def _build_reasons(
        self,
        ta_scores: dict,
        pattern_score: float,
        sentiment_score: float,
        patterns: PatternReport | None,
        sentiment: SentimentReport | None,
    ) -> list[str]:
        """Build human-readable reasons for the signal."""
        reasons = []

        trend = ta_scores["trend"]
        if trend > 20:
            reasons.append(f"Strong bullish trend (score: {trend:+.0f})")
        elif trend < -20:
            reasons.append(f"Strong bearish trend (score: {trend:+.0f})")
        else:
            reasons.append(f"Neutral/sideways trend (score: {trend:+.0f})")

        mom = ta_scores["momentum"]
        if mom > 15:
            reasons.append(f"Bullish momentum (RSI/Stoch oversold recovery)")
        elif mom < -15:
            reasons.append(f"Bearish momentum (RSI/Stoch overbought)")

        if patterns and patterns.patterns:
            top_patterns = sorted(patterns.patterns, key=lambda p: p.confidence, reverse=True)[:3]
            for p in top_patterns:
                reasons.append(f"Pattern: {p.description}")

        if sentiment and sentiment.fear_greed:
            fg = sentiment.fear_greed
            reasons.append(f"Market sentiment: {fg.label} ({fg.value}/100)")

        return reasons
