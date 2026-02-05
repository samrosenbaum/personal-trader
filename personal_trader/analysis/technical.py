"""Technical analysis indicators computed on OHLCV DataFrames.

Uses the `ta` library and raw numpy/pandas for custom calculations.
Every function takes a DataFrame with columns [open, high, low, close, volume]
and returns the same DataFrame with new indicator columns attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a comprehensive set of indicators to the candle DataFrame."""
    df = df.copy()
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_custom_indicators(df)
    return df


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Moving averages
    for period in (7, 20, 50, 100, 200):
        df[f"sma_{period}"] = ta.trend.sma_indicator(df["close"], window=period)
        df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], window=period)

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ADX
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
    df["ichimoku_a"] = ichi.ichimoku_a()
    df["ichimoku_b"] = ichi.ichimoku_b()
    df["ichimoku_base"] = ichi.ichimoku_base_line()
    df["ichimoku_conv"] = ichi.ichimoku_conversion_line()

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(df["high"], df["low"], df["close"])
    df["psar"] = psar.psar()
    df["psar_up"] = psar.psar_up()
    df["psar_down"] = psar.psar_down()

    return df


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"])

    # CCI
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

    # ROC (Rate of Change)
    df["roc"] = ta.momentum.roc(df["close"])

    # MFI (Money Flow Index)
    df["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"])

    return df


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    # ATR
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_middle"] = kc.keltner_channel_mband()

    # Donchian Channel
    dc = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"])
    df["dc_upper"] = dc.donchian_channel_hband()
    df["dc_lower"] = dc.donchian_channel_lband()

    return df


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # OBV
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # VWAP (session-based, approximated for crypto with rolling window)
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df[
        "volume"
    ].cumsum()

    # Volume SMA
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]

    # Accumulation/Distribution
    df["ad"] = ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"])

    # Chaikin Money Flow
    df["cmf"] = ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])

    return df


# ---------------------------------------------------------------------------
# Custom / Derived
# ---------------------------------------------------------------------------

def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Custom indicators useful for crypto-specific analysis."""

    # Squeeze detection (BB inside Keltner = low volatility squeeze)
    if all(c in df.columns for c in ("bb_upper", "bb_lower", "kc_upper", "kc_lower")):
        df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])

    # Higher-highs / lower-lows detection (5 period lookback)
    df["higher_high"] = df["high"] > df["high"].shift(1)
    df["lower_low"] = df["low"] < df["low"].shift(1)
    df["hh_count"] = df["higher_high"].rolling(5).sum()
    df["ll_count"] = df["lower_low"].rolling(5).sum()

    # Price distance from key MAs (percentage)
    for period in (50, 200):
        sma_col = f"sma_{period}"
        if sma_col in df.columns:
            df[f"dist_sma_{period}_pct"] = ((df["close"] - df[sma_col]) / df[sma_col]) * 100

    # Golden/Death cross signals
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]) & (
            df["sma_50"].shift(1) <= df["sma_200"].shift(1)
        )
        df["death_cross"] = (df["sma_50"] < df["sma_200"]) & (
            df["sma_50"].shift(1) >= df["sma_200"].shift(1)
        )

    # Candle body ratio (for doji / hammer detection assist)
    body = abs(df["close"] - df["open"])
    full_range = df["high"] - df["low"]
    df["body_ratio"] = np.where(full_range > 0, body / full_range, 0)

    return df


def compute_multi_timeframe(
    fetch_fn,
    symbol: str,
    timeframes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run indicators on multiple timeframes for the same symbol.

    Args:
        fetch_fn: Callable(symbol, timeframe, limit) -> pd.DataFrame of OHLCV
        symbol: Trading pair
        timeframes: List of timeframes to analyse

    Returns:
        Dict mapping timeframe -> indicator DataFrame
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h", "1d"]

    results = {}
    for tf in timeframes:
        try:
            df = fetch_fn(symbol, tf, 500)
            results[tf] = add_all_indicators(df)
        except Exception:
            pass
    return results
