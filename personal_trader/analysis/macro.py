"""Macro regime and cross-asset context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import pandas as pd
import requests


@dataclass
class MacroSnapshot:
    risk_on: bool
    dxy_trend: str
    spx_trend: str
    vix_level: float | None
    rates_level: float | None
    timestamp: datetime
    notes: list[str]


def spx_correlation(crypto_df: pd.DataFrame) -> float | None:
    """Compute rolling correlation of crypto vs SPX returns."""
    spx = _fetch_stooq("^spx")
    if spx is None or crypto_df is None or crypto_df.empty:
        return None
    spx_returns = spx["close"].pct_change().dropna()
    crypto_returns = crypto_df["close"].pct_change().dropna()
    aligned = pd.concat([spx_returns, crypto_returns], axis=1, join="inner")
    if aligned.empty:
        return None
    return float(aligned.corr().iloc[0, 1])


STOOQ_BASE = "https://stooq.pl/q/d/l/"


def _fetch_stooq(symbol: str, limit: int = 200) -> pd.DataFrame | None:
    try:
        resp = requests.get(STOOQ_BASE, params={"s": symbol, "i": "d"}, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df = df.rename(columns=str.lower)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df.tail(limit)
    except Exception:
        return None


def _trend_label(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    if df is None or df.empty or len(df) < slow:
        return "unknown"
    close = df["close"]
    fast_ma = close.rolling(fast).mean().iloc[-1]
    slow_ma = close.rolling(slow).mean().iloc[-1]
    if fast_ma > slow_ma:
        return "up"
    if fast_ma < slow_ma:
        return "down"
    return "flat"


def get_macro_snapshot() -> MacroSnapshot | None:
    """Fetch macro regime context using public market data."""
    dxy = _fetch_stooq("^dxy")
    spx = _fetch_stooq("^spx")
    vix = _fetch_stooq("^vix")
    rates = _fetch_stooq("^us10y")

    if not dxy or not spx:
        return None

    dxy_trend = _trend_label(dxy)
    spx_trend = _trend_label(spx)
    vix_level = float(vix["close"].iloc[-1]) if vix is not None and not vix.empty else None
    rates_level = float(rates["close"].iloc[-1]) if rates is not None and not rates.empty else None

    notes = []
    if vix_level and vix_level > 25:
        notes.append(f"VIX elevated ({vix_level:.1f})")
    if dxy_trend == "up":
        notes.append("USD strength headwind for risk assets")
    if spx_trend == "down":
        notes.append("Equity trend risk-off")

    risk_on = spx_trend == "up" and dxy_trend != "up" and (vix_level is None or vix_level < 25)

    return MacroSnapshot(
        risk_on=risk_on,
        dxy_trend=dxy_trend,
        spx_trend=spx_trend,
        vix_level=vix_level,
        rates_level=rates_level,
        timestamp=datetime.utcnow(),
        notes=notes,
    )
