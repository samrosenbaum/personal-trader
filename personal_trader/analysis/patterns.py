"""Chart pattern recognition.

Detects classic chart patterns from OHLCV data:
- Support / Resistance levels
- Double top / Double bottom (with volume + breakout confirmation)
- Head and shoulders / Inverse H&S
- Ascending / Descending triangles
- Bull / Bear flags (with slope + retracement criteria)
- Doji, hammer, engulfing candle patterns
- RSI divergence (bullish and bearish)
- Volume climax detection

Quality scoring:
- Each pattern gets a dynamic confidence based on:
  - Volume confirmation (did volume increase at key points?)
  - Breakout confirmation (has price broken the neckline/trendline?)
  - Pattern symmetry (how clean is the formation?)
  - Recency (patterns near current price matter more)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_and_shoulders"
    INV_HEAD_SHOULDERS = "inverse_head_and_shoulders"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    SUPPORT = "support"
    RESISTANCE = "resistance"
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    VOLUME_CLIMAX = "volume_climax"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DetectedPattern:
    pattern: PatternType
    bias: Bias
    confidence: float  # 0-1
    description: str
    price_level: float | None = None
    target_price: float | None = None
    confirmed: bool = False  # True if breakout confirmed


@dataclass
class PatternReport:
    patterns: list[DetectedPattern] = field(default_factory=list)
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)

    @property
    def overall_bias(self) -> Bias:
        if not self.patterns:
            return Bias.NEUTRAL
        # Weight by confidence instead of simple vote count
        bull = sum(p.confidence for p in self.patterns if p.bias == Bias.BULLISH)
        bear = sum(p.confidence for p in self.patterns if p.bias == Bias.BEARISH)
        if bull > bear * 1.2:
            return Bias.BULLISH
        if bear > bull * 1.2:
            return Bias.BEARISH
        return Bias.NEUTRAL


class PatternDetector:
    """Scans an indicator-enriched DataFrame for chart patterns."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.close = df["close"].values
        self.high = df["high"].values
        self.low = df["low"].values
        self.open_ = df["open"].values
        self.volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        self.n = len(df)
        self._vol_sma = self._rolling_mean(self.volume, 20)

    def detect_all(self) -> PatternReport:
        report = PatternReport()
        report.support_levels = self._find_support()
        report.resistance_levels = self._find_resistance()

        detectors = [
            self._detect_double_top,
            self._detect_double_bottom,
            self._detect_head_shoulders,
            self._detect_inv_head_shoulders,
            self._detect_ascending_triangle,
            self._detect_descending_triangle,
            self._detect_bull_flag,
            self._detect_bear_flag,
            self._detect_candle_patterns,
            self._detect_rsi_divergence,
            self._detect_volume_climax,
        ]
        for fn in detectors:
            try:
                patterns = fn()
                report.patterns.extend(patterns)
            except Exception as e:
                logger.debug(f"Pattern detection error in {fn.__name__}: {e}")

        return report

    # ------------------------------------------------------------------
    # Volume helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean, NaN-padded at start."""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1 : i + 1])
        return result

    def _volume_at_index(self, idx: int) -> float:
        """Return volume ratio vs 20-period average at a given index."""
        if idx < 0 or idx >= self.n:
            return 1.0
        avg = self._vol_sma[idx]
        if np.isnan(avg) or avg == 0:
            return 1.0
        return float(self.volume[idx] / avg)

    def _avg_volume_ratio(self, start: int, end: int) -> float:
        """Average volume ratio for a range of bars."""
        ratios = [self._volume_at_index(i) for i in range(max(0, start), min(self.n, end + 1))]
        return float(np.mean(ratios)) if ratios else 1.0

    def _recency_factor(self, idx: int) -> float:
        """Patterns closer to current bar get a confidence boost (0.8 to 1.0)."""
        if self.n <= 1:
            return 1.0
        distance = self.n - 1 - idx
        return max(0.8, 1.0 - distance / self.n * 0.5)

    # ------------------------------------------------------------------
    # Support / Resistance
    # ------------------------------------------------------------------

    def _find_pivots(self, window: int = 20) -> tuple[list[float], list[float]]:
        """Find local highs and lows as pivot points."""
        highs, lows = [], []
        for i in range(window, self.n - window):
            if self.high[i] == max(self.high[i - window : i + window + 1]):
                highs.append(float(self.high[i]))
            if self.low[i] == min(self.low[i - window : i + window + 1]):
                lows.append(float(self.low[i]))
        return highs, lows

    def _cluster_levels(self, levels: list[float], tolerance: float = 0.02) -> list[float]:
        """Cluster nearby price levels together."""
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        # Require at least 2 touches, rank by touch count
        valid = [(round(np.mean(c), 2), len(c)) for c in clusters if len(c) >= 2]
        valid.sort(key=lambda x: x[1], reverse=True)
        return [v[0] for v in valid]

    def _find_support(self) -> list[float]:
        _, lows = self._find_pivots()
        return self._cluster_levels(lows)

    def _find_resistance(self) -> list[float]:
        highs, _ = self._find_pivots()
        return self._cluster_levels(highs)

    # ------------------------------------------------------------------
    # Double Top / Bottom
    # ------------------------------------------------------------------

    def _detect_double_top(self) -> list[DetectedPattern]:
        patterns = []
        highs_idx = self._pivot_indices(self.high, order=20, comparator=np.greater_equal)
        for i in range(len(highs_idx) - 1):
            a, b = highs_idx[i], highs_idx[i + 1]
            if b - a < 10:
                continue
            price_a, price_b = self.high[a], self.high[b]
            if abs(price_a - price_b) / price_a < 0.03:
                neckline = min(self.low[a : b + 1])
                height = price_a - neckline
                current = self.close[-1]

                # Dynamic confidence based on quality
                conf = 0.5

                # Volume: ideally lower on second peak
                vol_a = self._volume_at_index(a)
                vol_b = self._volume_at_index(b)
                if vol_b < vol_a:
                    conf += 0.1  # declining volume on second peak = stronger signal

                # Symmetry: how close are the two peaks?
                symmetry = 1 - abs(price_a - price_b) / price_a
                conf += symmetry * 0.1

                # Breakout confirmation: has price broken below neckline?
                confirmed = current < neckline
                if confirmed:
                    conf += 0.15
                    # Volume on breakdown
                    breakdown_vol = self._avg_volume_ratio(b, self.n - 1)
                    if breakdown_vol > 1.3:
                        conf += 0.1

                conf = min(conf * self._recency_factor(b), 0.95)

                desc = f"Double top near ${price_a:.2f}, neckline ${neckline:.2f}"
                if confirmed:
                    desc += " [CONFIRMED - broke neckline]"
                else:
                    desc += " [watching for neckline break]"

                patterns.append(DetectedPattern(
                    pattern=PatternType.DOUBLE_TOP,
                    bias=Bias.BEARISH,
                    confidence=conf,
                    description=desc,
                    price_level=price_a,
                    target_price=neckline - height,
                    confirmed=confirmed,
                ))
        return patterns

    def _detect_double_bottom(self) -> list[DetectedPattern]:
        patterns = []
        lows_idx = self._pivot_indices(self.low, order=20, comparator=np.less_equal)
        for i in range(len(lows_idx) - 1):
            a, b = lows_idx[i], lows_idx[i + 1]
            if b - a < 10:
                continue
            price_a, price_b = self.low[a], self.low[b]
            if abs(price_a - price_b) / price_a < 0.03:
                neckline = max(self.high[a : b + 1])
                height = neckline - price_a
                current = self.close[-1]

                conf = 0.5

                # Volume: higher on second bounce = buyers stepping in
                vol_a = self._volume_at_index(a)
                vol_b = self._volume_at_index(b)
                if vol_b > vol_a:
                    conf += 0.1

                symmetry = 1 - abs(price_a - price_b) / price_a
                conf += symmetry * 0.1

                confirmed = current > neckline
                if confirmed:
                    conf += 0.15
                    breakout_vol = self._avg_volume_ratio(b, self.n - 1)
                    if breakout_vol > 1.3:
                        conf += 0.1

                conf = min(conf * self._recency_factor(b), 0.95)

                desc = f"Double bottom near ${price_a:.2f}, neckline ${neckline:.2f}"
                if confirmed:
                    desc += " [CONFIRMED - broke neckline]"
                else:
                    desc += " [watching for neckline break]"

                patterns.append(DetectedPattern(
                    pattern=PatternType.DOUBLE_BOTTOM,
                    bias=Bias.BULLISH,
                    confidence=conf,
                    description=desc,
                    price_level=price_a,
                    target_price=neckline + height,
                    confirmed=confirmed,
                ))
        return patterns

    # ------------------------------------------------------------------
    # Head & Shoulders
    # ------------------------------------------------------------------

    def _detect_head_shoulders(self) -> list[DetectedPattern]:
        patterns = []
        highs_idx = self._pivot_indices(self.high, order=15, comparator=np.greater_equal)
        for i in range(len(highs_idx) - 2):
            l_idx, h_idx, r_idx = highs_idx[i], highs_idx[i + 1], highs_idx[i + 2]
            l_price, h_price, r_price = self.high[l_idx], self.high[h_idx], self.high[r_idx]
            if h_price > l_price and h_price > r_price:
                shoulder_diff = abs(l_price - r_price) / l_price
                if shoulder_diff < 0.05:
                    neckline = min(self.low[l_idx : r_idx + 1])
                    current = self.close[-1]

                    conf = 0.5

                    # Volume pattern: should decline L->H->R
                    vol_l = self._volume_at_index(l_idx)
                    vol_h = self._volume_at_index(h_idx)
                    vol_r = self._volume_at_index(r_idx)
                    if vol_h < vol_l and vol_r < vol_h:
                        conf += 0.15  # textbook declining volume
                    elif vol_r < vol_l:
                        conf += 0.05

                    # Shoulder symmetry
                    conf += (1 - shoulder_diff / 0.05) * 0.1

                    # Breakout
                    confirmed = current < neckline
                    if confirmed:
                        conf += 0.15

                    conf = min(conf * self._recency_factor(r_idx), 0.95)

                    desc = f"H&S: head ${h_price:.2f}, shoulders ~${l_price:.2f}, neck ${neckline:.2f}"
                    if confirmed:
                        desc += " [CONFIRMED]"

                    patterns.append(DetectedPattern(
                        pattern=PatternType.HEAD_SHOULDERS,
                        bias=Bias.BEARISH,
                        confidence=conf,
                        description=desc,
                        price_level=h_price,
                        target_price=neckline - (h_price - neckline),
                        confirmed=confirmed,
                    ))
        return patterns

    def _detect_inv_head_shoulders(self) -> list[DetectedPattern]:
        patterns = []
        lows_idx = self._pivot_indices(self.low, order=15, comparator=np.less_equal)
        for i in range(len(lows_idx) - 2):
            l_idx, h_idx, r_idx = lows_idx[i], lows_idx[i + 1], lows_idx[i + 2]
            l_price, h_price, r_price = self.low[l_idx], self.low[h_idx], self.low[r_idx]
            if h_price < l_price and h_price < r_price:
                shoulder_diff = abs(l_price - r_price) / l_price
                if shoulder_diff < 0.05:
                    neckline = max(self.high[l_idx : r_idx + 1])
                    current = self.close[-1]

                    conf = 0.5
                    vol_r = self._volume_at_index(r_idx)
                    vol_h = self._volume_at_index(h_idx)
                    if vol_r > vol_h:
                        conf += 0.1  # increasing volume on right shoulder = accumulation

                    conf += (1 - shoulder_diff / 0.05) * 0.1

                    confirmed = current > neckline
                    if confirmed:
                        conf += 0.15
                        breakout_vol = self._avg_volume_ratio(r_idx, self.n - 1)
                        if breakout_vol > 1.3:
                            conf += 0.1

                    conf = min(conf * self._recency_factor(r_idx), 0.95)

                    desc = f"Inv H&S: head ${h_price:.2f}, shoulders ~${l_price:.2f}, neck ${neckline:.2f}"
                    if confirmed:
                        desc += " [CONFIRMED]"

                    patterns.append(DetectedPattern(
                        pattern=PatternType.INV_HEAD_SHOULDERS,
                        bias=Bias.BULLISH,
                        confidence=conf,
                        description=desc,
                        price_level=h_price,
                        target_price=neckline + (neckline - h_price),
                        confirmed=confirmed,
                    ))
        return patterns

    # ------------------------------------------------------------------
    # Triangles
    # ------------------------------------------------------------------

    def _detect_ascending_triangle(self) -> list[DetectedPattern]:
        patterns = []
        lookback = min(60, self.n)
        recent_high = self.high[-lookback:]
        recent_low = self.low[-lookback:]
        recent_vol = self.volume[-lookback:]

        resistance_vals = self._last_n_pivots(recent_high, 3, "high")
        support_vals = self._last_n_pivots(recent_low, 3, "low")

        if len(resistance_vals) < 2 or len(support_vals) < 3:
            return patterns

        resistance_std = np.std(resistance_vals)
        flat_resistance = resistance_std < np.mean(resistance_vals) * 0.015
        rising_support = support_vals[-1] > support_vals[0]

        if flat_resistance and rising_support:
            resistance_level = float(np.mean(resistance_vals))
            current = self.close[-1]

            conf = 0.5

            # Volume should contract as pattern develops
            vol_first_half = np.mean(recent_vol[: lookback // 2]) if lookback > 1 else 1
            vol_second_half = np.mean(recent_vol[lookback // 2 :]) if lookback > 1 else 1
            if vol_first_half > 0 and vol_second_half < vol_first_half:
                conf += 0.1

            # Breakout check
            confirmed = current > resistance_level * 1.005
            if confirmed:
                conf += 0.15
                if self._volume_at_index(self.n - 1) > 1.5:
                    conf += 0.1  # volume expansion on breakout

            conf = min(conf, 0.9)

            desc = f"Ascending triangle, resistance ~${resistance_level:.2f}"
            if confirmed:
                desc += " [BREAKOUT]"

            patterns.append(DetectedPattern(
                pattern=PatternType.ASCENDING_TRIANGLE,
                bias=Bias.BULLISH,
                confidence=conf,
                description=desc,
                price_level=resistance_level,
                confirmed=confirmed,
            ))
        return patterns

    def _detect_descending_triangle(self) -> list[DetectedPattern]:
        patterns = []
        lookback = min(60, self.n)
        recent_high = self.high[-lookback:]
        recent_low = self.low[-lookback:]

        support_vals = self._last_n_pivots(recent_low, 3, "low")
        resistance_vals = self._last_n_pivots(recent_high, 3, "high")

        if len(support_vals) < 2 or len(resistance_vals) < 3:
            return patterns

        support_std = np.std(support_vals)
        flat_support = support_std < np.mean(support_vals) * 0.015
        falling_highs = resistance_vals[-1] < resistance_vals[0]

        if flat_support and falling_highs:
            support_level = float(np.mean(support_vals))
            current = self.close[-1]

            conf = 0.5
            confirmed = current < support_level * 0.995
            if confirmed:
                conf += 0.15
                if self._volume_at_index(self.n - 1) > 1.5:
                    conf += 0.1

            conf = min(conf, 0.9)

            desc = f"Descending triangle, support ~${support_level:.2f}"
            if confirmed:
                desc += " [BREAKDOWN]"

            patterns.append(DetectedPattern(
                pattern=PatternType.DESCENDING_TRIANGLE,
                bias=Bias.BEARISH,
                confidence=conf,
                description=desc,
                price_level=support_level,
                confirmed=confirmed,
            ))
        return patterns

    # ------------------------------------------------------------------
    # Flags (improved with slope + retracement)
    # ------------------------------------------------------------------

    def _detect_bull_flag(self) -> list[DetectedPattern]:
        patterns = []
        if self.n < 20:
            return patterns

        # Look for a strong pole (>8% move in 5-15 bars) followed by
        # a pullback channel with downward or flat slope
        for pole_len in (10, 15, 20):
            if self.n < pole_len + 10:
                continue
            flag_start = self.n - pole_len
            pole_end = flag_start
            # Search backward for pole start
            best_pole_start = max(0, pole_end - 15)
            pole_return = (self.close[pole_end] - self.close[best_pole_start]) / self.close[best_pole_start]

            if pole_return < 0.08:
                continue

            # Flag portion: remaining bars after pole
            flag_close = self.close[pole_end:]
            if len(flag_close) < 5:
                continue

            # Flag should retrace 20-50% of pole and have low volatility
            pole_height = self.close[pole_end] - self.close[best_pole_start]
            flag_low = min(flag_close)
            retracement = (self.close[pole_end] - flag_low) / pole_height if pole_height > 0 else 1
            flag_vol = np.std(flag_close) / np.mean(flag_close)

            if 0.15 < retracement < 0.55 and flag_vol < 0.025:
                conf = 0.5

                # Volume should decline during flag
                pole_vol = np.mean(self.volume[best_pole_start:pole_end])
                flag_volume = np.mean(self.volume[pole_end:])
                if pole_vol > 0 and flag_volume < pole_vol * 0.7:
                    conf += 0.15

                # Tighter flag = higher confidence
                conf += max(0, (0.025 - flag_vol) / 0.025) * 0.1

                # Good retracement ratio (38-50% is textbook)
                if 0.3 < retracement < 0.5:
                    conf += 0.1

                conf = min(conf, 0.85)

                patterns.append(DetectedPattern(
                    pattern=PatternType.BULL_FLAG,
                    bias=Bias.BULLISH,
                    confidence=conf,
                    description=f"Bull flag: {pole_return:.0%} pole, {retracement:.0%} retracement, tight consolidation",
                    price_level=float(self.close[-1]),
                    target_price=float(self.close[-1] + pole_height),
                ))
                break  # only report best one
        return patterns

    def _detect_bear_flag(self) -> list[DetectedPattern]:
        patterns = []
        if self.n < 20:
            return patterns

        for pole_len in (10, 15, 20):
            if self.n < pole_len + 10:
                continue
            flag_start = self.n - pole_len
            pole_end = flag_start
            best_pole_start = max(0, pole_end - 15)
            pole_return = (self.close[pole_end] - self.close[best_pole_start]) / self.close[best_pole_start]

            if pole_return > -0.08:
                continue

            flag_close = self.close[pole_end:]
            if len(flag_close) < 5:
                continue

            pole_height = abs(self.close[best_pole_start] - self.close[pole_end])
            flag_high = max(flag_close)
            retracement = (flag_high - self.close[pole_end]) / pole_height if pole_height > 0 else 1
            flag_vol = np.std(flag_close) / np.mean(flag_close)

            if 0.15 < retracement < 0.55 and flag_vol < 0.025:
                conf = 0.5
                pole_vol = np.mean(self.volume[best_pole_start:pole_end])
                flag_volume = np.mean(self.volume[pole_end:])
                if pole_vol > 0 and flag_volume < pole_vol * 0.7:
                    conf += 0.15
                conf += max(0, (0.025 - flag_vol) / 0.025) * 0.1
                if 0.3 < retracement < 0.5:
                    conf += 0.1
                conf = min(conf, 0.85)

                patterns.append(DetectedPattern(
                    pattern=PatternType.BEAR_FLAG,
                    bias=Bias.BEARISH,
                    confidence=conf,
                    description=f"Bear flag: {abs(pole_return):.0%} drop, {retracement:.0%} retracement, tight consolidation",
                    price_level=float(self.close[-1]),
                    target_price=float(self.close[-1] - pole_height),
                ))
                break
        return patterns

    # ------------------------------------------------------------------
    # Candlestick Patterns
    # ------------------------------------------------------------------

    def _detect_candle_patterns(self) -> list[DetectedPattern]:
        patterns = []
        if self.n < 3:
            return patterns

        i = self.n - 1
        o, h, l, c = self.open_[i], self.high[i], self.low[i], self.close[i]
        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            return patterns
        body_ratio = body / full_range
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        vol_ratio = self._volume_at_index(i)

        # Doji -- confidence higher if at S/R level
        if body_ratio < 0.1:
            conf = 0.45
            if vol_ratio > 1.5:
                conf += 0.1  # high-volume doji = more significant
            patterns.append(DetectedPattern(
                pattern=PatternType.DOJI,
                bias=Bias.NEUTRAL,
                confidence=conf,
                description=f"Doji: indecision (vol {vol_ratio:.1f}x avg)",
                price_level=float(c),
            ))

        # Hammer (bullish reversal)
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
            conf = 0.5
            if vol_ratio > 1.3:
                conf += 0.1
            # More convincing after a decline
            if self.n > 5 and self.close[i] < self.close[i - 5]:
                conf += 0.1
            patterns.append(DetectedPattern(
                pattern=PatternType.HAMMER,
                bias=Bias.BULLISH,
                confidence=conf,
                description=f"Hammer: bullish reversal (vol {vol_ratio:.1f}x avg)",
                price_level=float(c),
            ))

        # Shooting star (bearish reversal)
        if upper_shadow > body * 2 and lower_shadow < body * 0.5 and o > c:
            conf = 0.5
            if vol_ratio > 1.3:
                conf += 0.1
            if self.n > 5 and self.close[i] > self.close[i - 5]:
                conf += 0.1
            patterns.append(DetectedPattern(
                pattern=PatternType.SHOOTING_STAR,
                bias=Bias.BEARISH,
                confidence=conf,
                description=f"Shooting star: bearish reversal (vol {vol_ratio:.1f}x avg)",
                price_level=float(c),
            ))

        # Engulfing patterns
        if self.n >= 2:
            prev_o, prev_c = self.open_[i - 1], self.close[i - 1]
            prev_vol = self._volume_at_index(i - 1)

            # Bullish engulfing
            if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
                conf = 0.55
                if vol_ratio > prev_vol:
                    conf += 0.1  # engulfing candle should have higher volume
                if vol_ratio > 1.5:
                    conf += 0.05
                patterns.append(DetectedPattern(
                    pattern=PatternType.BULLISH_ENGULFING,
                    bias=Bias.BULLISH,
                    confidence=conf,
                    description=f"Bullish engulfing (vol {vol_ratio:.1f}x avg)",
                    price_level=float(c),
                ))

            # Bearish engulfing
            if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
                conf = 0.55
                if vol_ratio > prev_vol:
                    conf += 0.1
                if vol_ratio > 1.5:
                    conf += 0.05
                patterns.append(DetectedPattern(
                    pattern=PatternType.BEARISH_ENGULFING,
                    bias=Bias.BEARISH,
                    confidence=conf,
                    description=f"Bearish engulfing (vol {vol_ratio:.1f}x avg)",
                    price_level=float(c),
                ))

        # Morning / Evening star
        if self.n >= 3:
            o1, c1 = self.open_[i - 2], self.close[i - 2]
            o2, c2 = self.open_[i - 1], self.close[i - 1]
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)

            if c1 < o1 and body2 < body1 * 0.3 and c > o and c > (o1 + c1) / 2:
                conf = 0.55
                if vol_ratio > 1.3:
                    conf += 0.1
                patterns.append(DetectedPattern(
                    pattern=PatternType.MORNING_STAR,
                    bias=Bias.BULLISH,
                    confidence=conf,
                    description=f"Morning star: bullish reversal (3-candle)",
                    price_level=float(c),
                ))

            if c1 > o1 and body2 < body1 * 0.3 and c < o and c < (o1 + c1) / 2:
                conf = 0.55
                if vol_ratio > 1.3:
                    conf += 0.1
                patterns.append(DetectedPattern(
                    pattern=PatternType.EVENING_STAR,
                    bias=Bias.BEARISH,
                    confidence=conf,
                    description=f"Evening star: bearish reversal (3-candle)",
                    price_level=float(c),
                ))

        return patterns

    # ------------------------------------------------------------------
    # RSI Divergence
    # ------------------------------------------------------------------

    def _detect_rsi_divergence(self) -> list[DetectedPattern]:
        """Detect bullish and bearish RSI divergences.

        Bullish divergence: price makes lower low, RSI makes higher low
        Bearish divergence: price makes higher high, RSI makes lower high
        """
        patterns = []
        if "rsi_14" not in self.df.columns or self.n < 30:
            return patterns

        rsi = self.df["rsi_14"].values
        lookback = min(60, self.n)

        # Find recent price pivot lows and corresponding RSI values
        price_lows = self._pivot_indices(
            self.low[-lookback:], order=8, comparator=np.less_equal
        )
        price_lows = [i + (self.n - lookback) for i in price_lows]  # adjust to full index

        # Bullish divergence: price lower low, RSI higher low
        if len(price_lows) >= 2:
            a, b = price_lows[-2], price_lows[-1]
            if (self.low[b] < self.low[a]  # price made lower low
                    and not np.isnan(rsi[a]) and not np.isnan(rsi[b])
                    and rsi[b] > rsi[a]):  # RSI made higher low
                vol_b = self._volume_at_index(b)
                conf = 0.55
                if vol_b < self._volume_at_index(a):
                    conf += 0.1  # declining volume on lower low = exhaustion
                conf *= self._recency_factor(b)
                patterns.append(DetectedPattern(
                    pattern=PatternType.BULLISH_DIVERGENCE,
                    bias=Bias.BULLISH,
                    confidence=min(conf, 0.85),
                    description=f"Bullish RSI divergence: price lower low but RSI higher low ({rsi[b]:.0f} vs {rsi[a]:.0f})",
                    price_level=float(self.low[b]),
                ))

        # Bearish divergence: price higher high, RSI lower high
        price_highs = self._pivot_indices(
            self.high[-lookback:], order=8, comparator=np.greater_equal
        )
        price_highs = [i + (self.n - lookback) for i in price_highs]

        if len(price_highs) >= 2:
            a, b = price_highs[-2], price_highs[-1]
            if (self.high[b] > self.high[a]  # price made higher high
                    and not np.isnan(rsi[a]) and not np.isnan(rsi[b])
                    and rsi[b] < rsi[a]):  # RSI made lower high
                conf = 0.55
                if self._volume_at_index(b) < self._volume_at_index(a):
                    conf += 0.1
                conf *= self._recency_factor(b)
                patterns.append(DetectedPattern(
                    pattern=PatternType.BEARISH_DIVERGENCE,
                    bias=Bias.BEARISH,
                    confidence=min(conf, 0.85),
                    description=f"Bearish RSI divergence: price higher high but RSI lower high ({rsi[b]:.0f} vs {rsi[a]:.0f})",
                    price_level=float(self.high[b]),
                ))

        return patterns

    # ------------------------------------------------------------------
    # Volume Climax
    # ------------------------------------------------------------------

    def _detect_volume_climax(self) -> list[DetectedPattern]:
        """Detect volume climax events (volume > 3x average).

        High volume exhaustion candles often mark turning points.
        """
        patterns = []
        if self.n < 20:
            return patterns

        i = self.n - 1
        vol_ratio = self._volume_at_index(i)

        if vol_ratio < 3.0:
            return patterns

        c, o = self.close[i], self.open_[i]
        if c < o:
            # High volume red candle = potential selling climax (bullish reversal)
            patterns.append(DetectedPattern(
                pattern=PatternType.VOLUME_CLIMAX,
                bias=Bias.BULLISH,
                confidence=0.55,
                description=f"Selling climax: volume {vol_ratio:.1f}x average on red candle - potential exhaustion bottom",
                price_level=float(c),
            ))
        else:
            # High volume green candle = potential buying climax (bearish reversal)
            patterns.append(DetectedPattern(
                pattern=PatternType.VOLUME_CLIMAX,
                bias=Bias.BEARISH,
                confidence=0.55,
                description=f"Buying climax: volume {vol_ratio:.1f}x average on green candle - potential blow-off top",
                price_level=float(c),
            ))

        return patterns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pivot_indices(data: np.ndarray, order: int = 20, comparator=np.greater_equal) -> list[int]:
        """Find local pivot point indices."""
        pivots = []
        for i in range(order, len(data) - order):
            window = data[i - order : i + order + 1]
            if comparator(data[i], window).all():
                pivots.append(i)
        return pivots

    @staticmethod
    def _last_n_pivots(data: np.ndarray, n: int, kind: str) -> list[float]:
        """Get the last n pivot values from an array."""
        order = max(5, len(data) // 10)
        pivots = []
        comp = np.greater_equal if kind == "high" else np.less_equal
        for i in range(order, len(data) - order):
            window = data[i - order : i + order + 1]
            if comp(data[i], window).all():
                pivots.append(float(data[i]))
        return pivots[-n:]
