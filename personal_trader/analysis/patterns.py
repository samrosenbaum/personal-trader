"""Chart pattern recognition.

Detects classic chart patterns from OHLCV data:
- Support / Resistance levels
- Double top / Double bottom
- Head and shoulders / Inverse H&S
- Ascending / Descending triangles
- Bull / Bear flags
- Doji, hammer, engulfing candle patterns
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


@dataclass
class PatternReport:
    patterns: list[DetectedPattern] = field(default_factory=list)
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)

    @property
    def overall_bias(self) -> Bias:
        if not self.patterns:
            return Bias.NEUTRAL
        bull = sum(1 for p in self.patterns if p.bias == Bias.BULLISH)
        bear = sum(1 for p in self.patterns if p.bias == Bias.BEARISH)
        if bull > bear:
            return Bias.BULLISH
        if bear > bull:
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
        self.n = len(df)

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
        ]
        for fn in detectors:
            try:
                patterns = fn()
                report.patterns.extend(patterns)
            except Exception as e:
                logger.debug(f"Pattern detection error in {fn.__name__}: {e}")

        return report

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
        return [round(np.mean(c), 2) for c in clusters if len(c) >= 2]

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
                neckline = min(self.low[a:b+1])
                height = price_a - neckline
                patterns.append(DetectedPattern(
                    pattern=PatternType.DOUBLE_TOP,
                    bias=Bias.BEARISH,
                    confidence=0.7,
                    description=f"Double top near {price_a:.2f}, neckline at {neckline:.2f}",
                    price_level=price_a,
                    target_price=neckline - height,
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
                neckline = max(self.high[a:b+1])
                height = neckline - price_a
                patterns.append(DetectedPattern(
                    pattern=PatternType.DOUBLE_BOTTOM,
                    bias=Bias.BULLISH,
                    confidence=0.7,
                    description=f"Double bottom near {price_a:.2f}, neckline at {neckline:.2f}",
                    price_level=price_a,
                    target_price=neckline + height,
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
            # Head must be higher than both shoulders
            if h_price > l_price and h_price > r_price:
                # Shoulders roughly equal
                if abs(l_price - r_price) / l_price < 0.05:
                    neckline = min(self.low[l_idx:r_idx+1])
                    patterns.append(DetectedPattern(
                        pattern=PatternType.HEAD_SHOULDERS,
                        bias=Bias.BEARISH,
                        confidence=0.75,
                        description=f"Head & shoulders: head {h_price:.2f}, shoulders ~{l_price:.2f}",
                        price_level=h_price,
                        target_price=neckline - (h_price - neckline),
                    ))
        return patterns

    def _detect_inv_head_shoulders(self) -> list[DetectedPattern]:
        patterns = []
        lows_idx = self._pivot_indices(self.low, order=15, comparator=np.less_equal)
        for i in range(len(lows_idx) - 2):
            l_idx, h_idx, r_idx = lows_idx[i], lows_idx[i + 1], lows_idx[i + 2]
            l_price, h_price, r_price = self.low[l_idx], self.low[h_idx], self.low[r_idx]
            if h_price < l_price and h_price < r_price:
                if abs(l_price - r_price) / l_price < 0.05:
                    neckline = max(self.high[l_idx:r_idx+1])
                    patterns.append(DetectedPattern(
                        pattern=PatternType.INV_HEAD_SHOULDERS,
                        bias=Bias.BULLISH,
                        confidence=0.75,
                        description=f"Inverse H&S: head {h_price:.2f}, shoulders ~{l_price:.2f}",
                        price_level=h_price,
                        target_price=neckline + (neckline - h_price),
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

        # Flat resistance, rising support
        resistance_std = np.std(self._last_n_pivots(recent_high, 3, "high"))
        support_vals = self._last_n_pivots(recent_low, 3, "low")

        if len(support_vals) >= 3 and resistance_std < np.mean(recent_high[-20:]) * 0.01:
            if support_vals[-1] > support_vals[0]:
                patterns.append(DetectedPattern(
                    pattern=PatternType.ASCENDING_TRIANGLE,
                    bias=Bias.BULLISH,
                    confidence=0.65,
                    description="Ascending triangle: flat resistance with rising lows",
                    price_level=float(np.mean(recent_high[-5:])),
                ))
        return patterns

    def _detect_descending_triangle(self) -> list[DetectedPattern]:
        patterns = []
        lookback = min(60, self.n)
        recent_high = self.high[-lookback:]
        recent_low = self.low[-lookback:]

        support_std = np.std(self._last_n_pivots(recent_low, 3, "low"))
        resistance_vals = self._last_n_pivots(recent_high, 3, "high")

        if len(resistance_vals) >= 3 and support_std < np.mean(recent_low[-20:]) * 0.01:
            if resistance_vals[-1] < resistance_vals[0]:
                patterns.append(DetectedPattern(
                    pattern=PatternType.DESCENDING_TRIANGLE,
                    bias=Bias.BEARISH,
                    confidence=0.65,
                    description="Descending triangle: flat support with falling highs",
                    price_level=float(np.mean(recent_low[-5:])),
                ))
        return patterns

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------

    def _detect_bull_flag(self) -> list[DetectedPattern]:
        patterns = []
        lookback = min(40, self.n)
        recent = self.close[-lookback:]
        # Strong upward move in first half, consolidation in second half
        mid = lookback // 2
        first_half_return = (recent[mid] - recent[0]) / recent[0]
        second_half_std = np.std(recent[mid:]) / np.mean(recent[mid:])

        if first_half_return > 0.05 and second_half_std < 0.02:
            patterns.append(DetectedPattern(
                pattern=PatternType.BULL_FLAG,
                bias=Bias.BULLISH,
                confidence=0.6,
                description="Bull flag: strong rally followed by tight consolidation",
                price_level=float(recent[-1]),
            ))
        return patterns

    def _detect_bear_flag(self) -> list[DetectedPattern]:
        patterns = []
        lookback = min(40, self.n)
        recent = self.close[-lookback:]
        mid = lookback // 2
        first_half_return = (recent[mid] - recent[0]) / recent[0]
        second_half_std = np.std(recent[mid:]) / np.mean(recent[mid:])

        if first_half_return < -0.05 and second_half_std < 0.02:
            patterns.append(DetectedPattern(
                pattern=PatternType.BEAR_FLAG,
                bias=Bias.BEARISH,
                confidence=0.6,
                description="Bear flag: strong drop followed by tight consolidation",
                price_level=float(recent[-1]),
            ))
        return patterns

    # ------------------------------------------------------------------
    # Candlestick Patterns (last few candles)
    # ------------------------------------------------------------------

    def _detect_candle_patterns(self) -> list[DetectedPattern]:
        patterns = []
        if self.n < 3:
            return patterns

        # Check last candle
        i = self.n - 1
        o, h, l, c = self.open_[i], self.high[i], self.low[i], self.close[i]
        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            return patterns
        body_ratio = body / full_range
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        # Doji
        if body_ratio < 0.1:
            patterns.append(DetectedPattern(
                pattern=PatternType.DOJI,
                bias=Bias.NEUTRAL,
                confidence=0.6,
                description="Doji candle: indecision, potential reversal",
                price_level=float(c),
            ))

        # Hammer (bullish reversal)
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
            patterns.append(DetectedPattern(
                pattern=PatternType.HAMMER,
                bias=Bias.BULLISH,
                confidence=0.6,
                description="Hammer candle: bullish reversal signal",
                price_level=float(c),
            ))

        # Shooting star (bearish reversal)
        if upper_shadow > body * 2 and lower_shadow < body * 0.5 and o > c:
            patterns.append(DetectedPattern(
                pattern=PatternType.SHOOTING_STAR,
                bias=Bias.BEARISH,
                confidence=0.6,
                description="Shooting star: bearish reversal signal",
                price_level=float(c),
            ))

        # Engulfing patterns (last 2 candles)
        if self.n >= 2:
            prev_o, prev_c = self.open_[i - 1], self.close[i - 1]
            # Bullish engulfing
            if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
                patterns.append(DetectedPattern(
                    pattern=PatternType.BULLISH_ENGULFING,
                    bias=Bias.BULLISH,
                    confidence=0.7,
                    description="Bullish engulfing: strong reversal signal",
                    price_level=float(c),
                ))
            # Bearish engulfing
            if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
                patterns.append(DetectedPattern(
                    pattern=PatternType.BEARISH_ENGULFING,
                    bias=Bias.BEARISH,
                    confidence=0.7,
                    description="Bearish engulfing: strong reversal signal",
                    price_level=float(c),
                ))

        # Morning / Evening star (last 3 candles)
        if self.n >= 3:
            o1, c1 = self.open_[i - 2], self.close[i - 2]
            o2, c2 = self.open_[i - 1], self.close[i - 1]
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            body3 = body

            # Morning star
            if c1 < o1 and body2 < body1 * 0.3 and c > o and c > (o1 + c1) / 2:
                patterns.append(DetectedPattern(
                    pattern=PatternType.MORNING_STAR,
                    bias=Bias.BULLISH,
                    confidence=0.7,
                    description="Morning star: bullish reversal (3-candle pattern)",
                    price_level=float(c),
                ))

            # Evening star
            if c1 > o1 and body2 < body1 * 0.3 and c < o and c < (o1 + c1) / 2:
                patterns.append(DetectedPattern(
                    pattern=PatternType.EVENING_STAR,
                    bias=Bias.BEARISH,
                    confidence=0.7,
                    description="Evening star: bearish reversal (3-candle pattern)",
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
