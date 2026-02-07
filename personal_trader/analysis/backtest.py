"""Simple backtesting utilities for signal validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from personal_trader.analysis.technical import add_all_indicators
from personal_trader.strategy.signals import Action, SignalGenerator


@dataclass
class BacktestResult:
    symbol: str
    win_rate: float
    avg_r_multiple: float
    trades: int
    sample_size: int


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    risk_profile,
    lookahead: int = 10,
    min_history: int = 100,
) -> BacktestResult | None:
    """Backtest signals on a single timeframe.

    Uses the signal generator on rolling windows and evaluates whether
    stop or target was hit within a lookahead window.
    """
    if df is None or len(df) < min_history:
        return None

    df = add_all_indicators(df)
    gen = SignalGenerator(risk_profile)

    wins = 0
    losses = 0
    r_multiples = []
    trades = 0

    for idx in range(min_history, len(df) - lookahead):
        window = df.iloc[:idx]
        indicators = {"1d": window}
        signals = gen.generate(symbol=symbol, indicators=indicators)
        if not signals:
            continue
        sig = signals[0]
        if sig.action not in (Action.BUY, Action.STRONG_BUY):
            continue
        if sig.entry_price is None or sig.stop_loss is None or sig.take_profit is None:
            continue

        trades += 1
        future = df.iloc[idx: idx + lookahead]
        low = future["low"].min()
        high = future["high"].max()
        risk = sig.entry_price - sig.stop_loss
        reward = sig.take_profit - sig.entry_price
        if risk <= 0:
            continue

        if low <= sig.stop_loss:
            losses += 1
            r_multiples.append(-1)
        elif high >= sig.take_profit:
            wins += 1
            r_multiples.append(reward / risk)
        else:
            r_multiples.append(0)

    if trades == 0:
        return None

    win_rate = wins / trades if trades else 0
    avg_r = float(np.mean(r_multiples)) if r_multiples else 0

    return BacktestResult(
        symbol=symbol,
        win_rate=round(win_rate, 3),
        avg_r_multiple=round(avg_r, 2),
        trades=trades,
        sample_size=len(df),
    )
