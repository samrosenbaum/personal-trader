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

Signal intelligence:
- Uses support/resistance from pattern detection for stop/target placement
- Multi-timeframe confluence scoring (signals on multiple TFs = higher confidence)
- Market structure analysis (higher highs/lows or lower highs/lows)
- Momentum divergence awareness (price vs indicators disagreement)
- Volatility regime detection (compression vs expansion)
- Order flow inference from volume profile and price action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from personal_trader.analysis.derivatives_market import FuturesBasis, VolSurface
from personal_trader.analysis.macro import MacroSnapshot
from personal_trader.analysis.microstructure import OrderBookMetrics, VolumeProfile
from personal_trader.analysis.patterns import Bias, DetectedPattern, PatternReport, PatternType
from personal_trader.analysis.sentiment import SentimentLevel, SentimentReport
from personal_trader.config import RiskProfile
from personal_trader.strategy.orders import OrderPlan, OrderPlanner

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
    order_plan: OrderPlan | None = None

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
        self.order_planner = OrderPlanner(risk_profile)

    def generate(
        self,
        symbol: str,
        indicators: dict[str, pd.DataFrame],
        patterns: PatternReport | None = None,
        sentiment: SentimentReport | None = None,
        microstructure: OrderBookMetrics | None = None,
        volume_profile: VolumeProfile | None = None,
        anchored_vwap: float | None = None,
        macro: MacroSnapshot | None = None,
        vol_surface: VolSurface | None = None,
        futures_basis: FuturesBasis | None = None,
        cross_asset_corr: float | None = None,
        portfolio_value: float | None = None,
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

        # Advanced analysis layers
        confluence = self._multi_timeframe_confluence(indicators)
        structure = self._market_structure(indicators)
        vol_regime = self._volatility_regime(indicators)
        divergence_adj = self._divergence_adjustment(indicators, patterns)

        # Weighted composite score (-100 to +100)
        # Base weights adjusted by confluence strength
        weights = {"technical": 0.45, "pattern": 0.25, "sentiment": 0.20, "structure": 0.10}
        composite = (
            ta_scores["composite"] * weights["technical"]
            + pattern_score * weights["pattern"]
            + sentiment_score * weights["sentiment"]
            + structure["score"] * weights["structure"]
        )

        # Microstructure adjustment
        if microstructure:
            if microstructure.spread_pct > 0.2:
                composite *= 0.9
            if microstructure.bid_ask_imbalance > 0.15:
                composite += 5
            if microstructure.bid_ask_imbalance < -0.15:
                composite -= 5

        # Macro regime adjustment
        if macro:
            composite += 5 if macro.risk_on else -5

        # Derivatives bias adjustment
        if futures_basis:
            if futures_basis.basis_pct > 0.3:
                composite += 3
            elif futures_basis.basis_pct < -0.3:
                composite -= 3

        # Apply modifiers from advanced analysis
        # Confluence: if multiple timeframes agree, amplify the signal
        composite *= confluence["amplifier"]

        # Divergence: if price/indicator divergence detected, shift toward reversal
        composite += divergence_adj

        # Volatility regime: in compression, reduce confidence; in expansion, signals matter more
        vol_conf_modifier = vol_regime["confidence_modifier"]

        composite = max(-100, min(100, composite))

        # Determine primary action
        action, urgency, confidence = self._score_to_action(composite, ta_scores, sentiment)
        confidence = min(confidence * vol_conf_modifier, 0.95)

        reasons = self._build_reasons(
            ta_scores, pattern_score, sentiment_score, patterns, sentiment,
            confluence=confluence,
            structure=structure,
            vol_regime=vol_regime,
            microstructure=microstructure,
            volume_profile=volume_profile,
            anchored_vwap=anchored_vwap,
            macro=macro,
            vol_surface=vol_surface,
            futures_basis=futures_basis,
            cross_asset_corr=cross_asset_corr,
        )

        # Calculate entry/stop/target from S/R levels + technicals
        entry, stop, target, atr = self._compute_levels(indicators, action, patterns)

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
        primary.order_plan = self.order_planner.build_plan(primary, portfolio_value, atr=atr)
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
        self,
        indicators: dict[str, pd.DataFrame],
        action: Action,
        patterns: PatternReport | None = None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Compute entry, stop-loss, and take-profit using S/R levels + ATR.

        Priority for stop/target placement:
        1. Nearest support/resistance from pattern detection
        2. Key moving averages (50, 100, 200 SMA)
        3. Bollinger Band extremes
        4. ATR-based fallback
        """
        for tf in ("4h", "1h", "1d", "15m"):
            if tf not in indicators or indicators[tf].empty:
                continue
            df = indicators[tf]
            last = df.iloc[-1]
            close = float(last["close"])
            atr = float(last.get("atr_14", close * 0.02))

            if action not in (
                Action.BUY, Action.STRONG_BUY,
                Action.SELL, Action.STRONG_SELL, Action.MOVE_TO_STABLE,
            ):
                return close, None, None, None

            is_long = action in (Action.BUY, Action.STRONG_BUY)

            # Gather candidate levels from S/R, MAs, BBands
            support_levels = []
            resistance_levels = []

            if patterns:
                support_levels.extend(patterns.support_levels)
                resistance_levels.extend(patterns.resistance_levels)
                # Also use pattern target prices
                for p in patterns.patterns:
                    if p.target_price:
                        if p.target_price < close:
                            support_levels.append(p.target_price)
                        else:
                            resistance_levels.append(p.target_price)

            # Add moving averages as dynamic S/R
            for ma_col in ("sma_50", "sma_100", "sma_200", "ema_50", "ema_200"):
                if ma_col in df.columns:
                    val = last.get(ma_col)
                    if val and not pd.isna(val):
                        fval = float(val)
                        if fval < close:
                            support_levels.append(fval)
                        else:
                            resistance_levels.append(fval)

            # Add Bollinger Bands
            if "bb_lower" in df.columns and not pd.isna(last.get("bb_lower")):
                support_levels.append(float(last["bb_lower"]))
            if "bb_upper" in df.columns and not pd.isna(last.get("bb_upper")):
                resistance_levels.append(float(last["bb_upper"]))

            # Add VWAP
            if "vwap" in df.columns and not pd.isna(last.get("vwap")):
                vwap = float(last["vwap"])
                if vwap < close:
                    support_levels.append(vwap)
                else:
                    resistance_levels.append(vwap)

            # Filter and sort
            support_below = sorted([s for s in support_levels if s < close], reverse=True)
            resistance_above = sorted([r for r in resistance_levels if r > close])

            entry = close

            if is_long:
                # Stop below nearest support (with ATR buffer)
                if support_below:
                    stop = support_below[0] - atr * 0.5
                else:
                    stop = close - atr * 2

                # Target at nearest resistance (or 2nd if first is too close)
                if resistance_above:
                    target = resistance_above[0]
                    # If first resistance is <1 ATR away, aim for the next one
                    if target - close < atr and len(resistance_above) > 1:
                        target = resistance_above[1]
                else:
                    target = close + atr * 3
            else:
                # Short: stop above nearest resistance
                if resistance_above:
                    stop = resistance_above[0] + atr * 0.5
                else:
                    stop = close + atr * 2

                # Target at nearest support
                if support_below:
                    target = support_below[0]
                    if close - target < atr and len(support_below) > 1:
                        target = support_below[1]
                else:
                    target = close - atr * 3

            # Sanity: ensure minimum 1:1.5 risk:reward
            risk = abs(close - stop)
            reward = abs(target - close)
            if risk > 0 and reward / risk < 1.5:
                if is_long:
                    target = close + risk * 2
                else:
                    target = close - risk * 2

            return round(entry, 2), round(stop, 2), round(target, 2), round(atr, 4)

        return None, None, None, None

    # ------------------------------------------------------------------
    # Advanced analysis methods
    # ------------------------------------------------------------------

    def _multi_timeframe_confluence(self, indicators: dict[str, pd.DataFrame]) -> dict:
        """Check if multiple timeframes agree on direction.

        When 15m, 1h, 4h, and 1d all say the same thing, that's a much
        stronger signal than a single timeframe.
        """
        bullish_count = 0
        bearish_count = 0
        total = 0
        tf_signals = {}

        for tf, df in indicators.items():
            if df is None or df.empty:
                continue
            total += 1
            last = df.iloc[-1]
            direction = 0

            # MA alignment
            if "sma_20" in df.columns and "sma_50" in df.columns:
                sma20 = last.get("sma_20")
                sma50 = last.get("sma_50")
                close = last["close"]
                if sma20 and sma50 and not pd.isna(sma20) and not pd.isna(sma50):
                    if close > sma20 > sma50:
                        direction += 1
                    elif close < sma20 < sma50:
                        direction -= 1

            # MACD direction
            if "macd_hist" in df.columns:
                mh = last.get("macd_hist")
                if mh and not pd.isna(mh):
                    direction += 1 if mh > 0 else -1

            # RSI zone
            if "rsi_14" in df.columns:
                rsi = last.get("rsi_14")
                if rsi and not pd.isna(rsi):
                    if rsi > 55:
                        direction += 1
                    elif rsi < 45:
                        direction -= 1

            if direction >= 2:
                bullish_count += 1
                tf_signals[tf] = "bullish"
            elif direction <= -2:
                bearish_count += 1
                tf_signals[tf] = "bearish"
            else:
                tf_signals[tf] = "neutral"

        if total == 0:
            return {"amplifier": 1.0, "agreement": 0, "tf_signals": {}}

        max_agree = max(bullish_count, bearish_count)
        agreement_pct = max_agree / total

        # Amplify signal when TFs agree: 3/4 agree = 1.2x, 4/4 = 1.4x
        if agreement_pct >= 0.9:
            amplifier = 1.4
        elif agreement_pct >= 0.7:
            amplifier = 1.2
        elif agreement_pct >= 0.5:
            amplifier = 1.0
        else:
            amplifier = 0.85  # disagreement = reduce conviction

        return {
            "amplifier": amplifier,
            "agreement": agreement_pct,
            "bullish_tfs": bullish_count,
            "bearish_tfs": bearish_count,
            "tf_signals": tf_signals,
        }

    def _market_structure(self, indicators: dict[str, pd.DataFrame]) -> dict:
        """Analyze market structure: HH/HL (uptrend) vs LH/LL (downtrend).

        This goes beyond simple MA positioning to understand the swing
        structure of the market.
        """
        score = 0

        # Use 4h or 1d for structure (need enough history)
        for tf in ("4h", "1d"):
            if tf not in indicators or indicators[tf].empty:
                continue
            df = indicators[tf]
            if len(df) < 30:
                continue

            highs = df["high"].values
            lows = df["low"].values

            # Find swing highs and lows (local extremes with 5-bar window)
            swing_highs = []
            swing_lows = []
            order = 5
            for i in range(order, len(df) - order):
                if highs[i] == max(highs[i - order: i + order + 1]):
                    swing_highs.append(float(highs[i]))
                if lows[i] == min(lows[i - order: i + order + 1]):
                    swing_lows.append(float(lows[i]))

            if len(swing_highs) < 2 or len(swing_lows) < 2:
                continue

            # Check last 3 swing highs/lows for structure
            recent_highs = swing_highs[-3:]
            recent_lows = swing_lows[-3:]

            # Higher highs and higher lows = bullish structure
            hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
            # Lower highs and lower lows = bearish structure
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])

            if hh_count >= 1 and hl_count >= 1:
                score += 30  # bullish structure
            elif lh_count >= 1 and ll_count >= 1:
                score -= 30  # bearish structure

            # Break of structure detection
            current_price = float(df["close"].iloc[-1])
            if swing_lows and current_price < swing_lows[-1]:
                score -= 20  # broke below last swing low = structure break bearish
            elif swing_highs and current_price > swing_highs[-1]:
                score += 20  # broke above last swing high = structure break bullish

            break  # only use best timeframe

        return {"score": max(-100, min(100, score))}

    def _volatility_regime(self, indicators: dict[str, pd.DataFrame]) -> dict:
        """Detect if we're in a compression or expansion phase.

        Compression (low vol): big move coming, but direction uncertain - lower confidence
        Expansion (high vol): signals are playing out - higher confidence
        """
        for tf in ("4h", "1d", "1h"):
            if tf not in indicators or indicators[tf].empty:
                continue
            df = indicators[tf]
            last = df.iloc[-1]

            squeeze = False
            bb_width_percentile = 0.5
            regime = "normal"

            if "squeeze_on" in df.columns and last.get("squeeze_on"):
                squeeze = True

            if "bb_width" in df.columns and not pd.isna(last.get("bb_width")):
                bb_w = last["bb_width"]
                # Compare to rolling BB width
                bb_widths = df["bb_width"].dropna()
                if len(bb_widths) > 20:
                    bb_width_percentile = float(
                        (bb_widths < bb_w).sum() / len(bb_widths)
                    )

            if squeeze or bb_width_percentile < 0.2:
                regime = "compression"
                confidence_modifier = 0.85  # less confident, waiting for breakout
            elif bb_width_percentile > 0.8:
                regime = "expansion"
                confidence_modifier = 1.1  # vol expansion = signals matter more
            else:
                regime = "normal"
                confidence_modifier = 1.0

            return {
                "regime": regime,
                "confidence_modifier": confidence_modifier,
                "squeeze": squeeze,
                "bb_width_percentile": bb_width_percentile,
            }

        return {"regime": "unknown", "confidence_modifier": 1.0, "squeeze": False, "bb_width_percentile": 0.5}

    def _divergence_adjustment(
        self, indicators: dict[str, pd.DataFrame], patterns: PatternReport | None,
    ) -> float:
        """Score adjustments based on price/indicator divergences.

        Divergences (from pattern detection) and volume/price disagreements
        are early warnings of reversals.
        """
        adj = 0.0

        # Use divergences detected by pattern detector
        if patterns:
            for p in patterns.patterns:
                if p.pattern == PatternType.BULLISH_DIVERGENCE:
                    adj += 10 * p.confidence
                elif p.pattern == PatternType.BEARISH_DIVERGENCE:
                    adj -= 10 * p.confidence
                elif p.pattern == PatternType.VOLUME_CLIMAX:
                    # Volume climax = potential reversal
                    if p.bias == Bias.BULLISH:
                        adj += 8 * p.confidence
                    else:
                        adj -= 8 * p.confidence

        # Check for OBV divergence (price up but OBV down, or vice versa)
        for tf in ("4h", "1d"):
            if tf not in indicators or indicators[tf].empty:
                continue
            df = indicators[tf]
            if len(df) < 20 or "obv" not in df.columns:
                continue

            close_10 = df["close"].iloc[-10:]
            obv_10 = df["obv"].iloc[-10:]

            price_up = float(close_10.iloc[-1]) > float(close_10.iloc[0])
            obv_up = float(obv_10.iloc[-1]) > float(obv_10.iloc[0])

            if price_up and not obv_up:
                adj -= 5  # price rising on declining volume = weak
            elif not price_up and obv_up:
                adj += 5  # price falling but accumulation happening = bullish divergence
            break

        return max(-20, min(20, adj))

    # ------------------------------------------------------------------
    # Reason building
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        ta_scores: dict,
        pattern_score: float,
        sentiment_score: float,
        patterns: PatternReport | None,
        sentiment: SentimentReport | None,
        confluence: dict | None = None,
        structure: dict | None = None,
        vol_regime: dict | None = None,
        microstructure: OrderBookMetrics | None = None,
        volume_profile: VolumeProfile | None = None,
        anchored_vwap: float | None = None,
        macro: MacroSnapshot | None = None,
        vol_surface: VolSurface | None = None,
        futures_basis: FuturesBasis | None = None,
        cross_asset_corr: float | None = None,
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
            reasons.append("Bullish momentum (RSI/Stoch oversold recovery)")
        elif mom < -15:
            reasons.append("Bearish momentum (RSI/Stoch overbought)")

        # Multi-timeframe confluence
        if confluence and confluence.get("agreement", 0) >= 0.7:
            bullish = confluence.get("bullish_tfs", 0)
            bearish = confluence.get("bearish_tfs", 0)
            if bullish > bearish:
                reasons.append(f"Multi-timeframe confluence: {bullish} timeframes bullish")
            elif bearish > bullish:
                reasons.append(f"Multi-timeframe confluence: {bearish} timeframes bearish")

        # Market structure
        if structure and abs(structure.get("score", 0)) >= 20:
            if structure["score"] > 0:
                reasons.append("Market structure: higher highs and higher lows (uptrend)")
            else:
                reasons.append("Market structure: lower highs and lower lows (downtrend)")

        # Volatility regime
        if vol_regime:
            regime = vol_regime.get("regime", "")
            if regime == "compression":
                reasons.append("Volatility compression (squeeze) - big move building")
            elif regime == "expansion":
                reasons.append("Volatility expanding - trend in motion")

        if patterns and patterns.patterns:
            top_patterns = sorted(patterns.patterns, key=lambda p: p.confidence, reverse=True)[:3]
            for p in top_patterns:
                tag = " [CONFIRMED]" if p.confirmed else ""
                reasons.append(f"Pattern: {p.description}{tag}")

        if sentiment and sentiment.fear_greed:
            fg = sentiment.fear_greed
            reasons.append(f"Market sentiment: {fg.label} ({fg.value}/100)")

        if microstructure:
            reasons.append(
                f"Order book: spread {microstructure.spread_pct:.2f}% "
                f"| imbalance {microstructure.bid_ask_imbalance:+.2f}"
            )

        if volume_profile:
            reasons.append(
                f"Volume profile POC {volume_profile.poc:.2f} "
                f"(VA {volume_profile.value_area_low:.2f}-{volume_profile.value_area_high:.2f})"
            )

        if anchored_vwap:
            reasons.append(f"Anchored VWAP: {anchored_vwap:.2f}")

        if macro:
            regime = "risk-on" if macro.risk_on else "risk-off"
            reasons.append(
                f"Macro regime: {regime} | DXY {macro.dxy_trend}, SPX {macro.spx_trend}"
            )
            if macro.notes:
                reasons.append(f"Macro notes: {', '.join(macro.notes[:2])}")

        if vol_surface and vol_surface.points:
            point = vol_surface.points[0]
            if point.atm_iv:
                reasons.append(f"Options IV {point.tenor_days}d: {point.atm_iv:.1f}%")
            if point.skew is not None:
                reasons.append(f"IV skew {point.tenor_days}d: {point.skew:+.1f}")

        if futures_basis:
            reasons.append(f"Futures basis: {futures_basis.basis_pct:+.2f}%")

        if cross_asset_corr is not None:
            reasons.append(f"SPX correlation: {cross_asset_corr:+.2f}")

        return reasons
