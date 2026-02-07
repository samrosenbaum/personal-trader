"""Market microstructure analytics for execution-aware signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrderBookMetrics:
    mid_price: float
    spread_pct: float
    bid_ask_imbalance: float
    bid_depth: float
    ask_depth: float
    depth_ratio: float
    slippage_1pct: float


@dataclass
class VolumeProfile:
    poc: float  # point of control
    value_area_low: float
    value_area_high: float
    bins: dict[float, float]


def analyze_order_book(order_book: dict, depth: int = 25) -> OrderBookMetrics | None:
    """Compute order book imbalance, spread, and depth ratios."""
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids or not asks:
        return None

    bids = bids[:depth]
    asks = asks[:depth]

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2
    spread_pct = ((best_ask - best_bid) / mid) * 100 if mid else 0.0

    bid_depth = sum(float(p) * float(sz) for p, sz in bids)
    ask_depth = sum(float(p) * float(sz) for p, sz in asks)
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0.0
    depth_ratio = (bid_depth / ask_depth) if ask_depth else 0.0

    # Rough slippage estimate: 1% move cost based on depth
    slippage_1pct = (mid * 0.01) / (total_depth / mid) if mid and total_depth else 0.0

    return OrderBookMetrics(
        mid_price=mid,
        spread_pct=round(spread_pct, 4),
        bid_ask_imbalance=round(imbalance, 4),
        bid_depth=round(bid_depth, 2),
        ask_depth=round(ask_depth, 2),
        depth_ratio=round(depth_ratio, 2),
        slippage_1pct=round(slippage_1pct, 6),
    )


def volume_profile(df: pd.DataFrame, bins: int = 24) -> VolumeProfile | None:
    """Create a simple volume profile across price bins."""
    if df is None or df.empty:
        return None
    prices = df["close"].values
    volumes = df["volume"].values
    if len(prices) < bins:
        return None

    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    if min_price == max_price:
        return None

    edges = np.linspace(min_price, max_price, bins + 1)
    digitized = np.digitize(prices, edges) - 1
    profile = {}
    for idx, vol in zip(digitized, volumes, strict=False):
        if 0 <= idx < bins:
            price_level = float((edges[idx] + edges[idx + 1]) / 2)
            profile[price_level] = profile.get(price_level, 0.0) + float(vol)

    if not profile:
        return None

    sorted_profile = dict(sorted(profile.items(), key=lambda x: x[1], reverse=True))
    poc = next(iter(sorted_profile))
    total_vol = sum(sorted_profile.values())
    cumulative = 0.0
    value_area = []
    for price, vol in sorted_profile.items():
        cumulative += vol
        value_area.append(price)
        if cumulative >= total_vol * 0.7:
            break

    return VolumeProfile(
        poc=round(poc, 2),
        value_area_low=round(min(value_area), 2),
        value_area_high=round(max(value_area), 2),
        bins=sorted_profile,
    )


def anchored_vwap(df: pd.DataFrame, anchor_window: int = 50) -> float | None:
    """Compute anchored VWAP from the most recent swing low."""
    if df is None or df.empty or len(df) < anchor_window:
        return None
    recent = df.tail(anchor_window)
    anchor_idx = recent["low"].idxmin()
    anchor_df = df.loc[anchor_idx:]
    if anchor_df.empty:
        return None

    price = (anchor_df["high"] + anchor_df["low"] + anchor_df["close"]) / 3
    vwap = (price * anchor_df["volume"]).sum() / anchor_df["volume"].sum()
    return round(float(vwap), 2)
