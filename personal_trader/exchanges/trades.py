"""Trade history, cost basis, and realized P&L tracking.

Provides unified data structures for trades from all sources (Robinhood,
Coinbase, Binance, etc.) and FIFO-based P&L calculations.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class Trade:
    """A single executed trade, normalized across all sources."""

    id: str
    symbol: str  # base asset, e.g. "BTC", "ETH"
    side: TradeSide
    quantity: float
    price: float  # execution price per unit
    total: float  # quantity * price (USD value)
    fee: float  # fees in USD
    timestamp: datetime
    source: str  # "robinhood", "coinbase", "binance", etc.
    asset_type: str  # "crypto", "stock", "etf"
    status: TradeStatus = TradeStatus.FILLED
    order_id: str = ""
    pair: str = ""  # e.g. "BTC/USDT" for exchange trades


@dataclass
class RealizedPnL:
    """Realized profit/loss for a symbol."""

    symbol: str
    total_realized_pnl: float
    total_cost_basis: float  # total $ spent on buys that were later sold
    total_proceeds: float  # total $ received from sells
    total_fees: float
    trade_count: int  # number of round-trip closes
    avg_buy_price: float
    avg_sell_price: float
    pnl_pct: float  # total_realized_pnl / total_cost_basis * 100
    first_trade: datetime | None = None
    last_trade: datetime | None = None


@dataclass
class TradeHistory:
    """Complete trade history across all sources."""

    trades: list[Trade] = field(default_factory=list)
    realized_pnl: dict[str, RealizedPnL] = field(default_factory=dict)
    total_realized_pnl: float = 0.0
    total_fees: float = 0.0

    @property
    def by_symbol(self) -> dict[str, list[Trade]]:
        result: dict[str, list[Trade]] = {}
        for t in self.trades:
            result.setdefault(t.symbol, []).append(t)
        return result

    @property
    def by_source(self) -> dict[str, list[Trade]]:
        result: dict[str, list[Trade]] = {}
        for t in self.trades:
            result.setdefault(t.source, []).append(t)
        return result

    @property
    def buys(self) -> list[Trade]:
        return [t for t in self.trades if t.side == TradeSide.BUY]

    @property
    def sells(self) -> list[Trade]:
        return [t for t in self.trades if t.side == TradeSide.SELL]


def compute_realized_pnl(trades: list[Trade]) -> dict[str, RealizedPnL]:
    """Calculate realized P&L per symbol using FIFO cost basis.

    For each symbol, sorts trades chronologically, maintains a FIFO queue
    of buy lots, and matches sells against the oldest buys first.
    """
    # Group by symbol
    by_symbol: dict[str, list[Trade]] = {}
    for t in trades:
        if t.status != TradeStatus.FILLED:
            continue
        by_symbol.setdefault(t.symbol, []).append(t)

    results: dict[str, RealizedPnL] = {}

    for symbol, symbol_trades in by_symbol.items():
        sorted_trades = sorted(symbol_trades, key=lambda t: t.timestamp)

        # FIFO queue of (remaining_qty, price_per_unit)
        buy_lots: deque[list[float]] = deque()

        total_cost_basis = 0.0
        total_proceeds = 0.0
        total_fees = 0.0
        close_count = 0
        total_buy_qty_sold = 0.0  # for avg buy price of sold lots
        total_buy_cost_sold = 0.0
        total_sell_qty = 0.0
        total_sell_proceeds = 0.0

        for trade in sorted_trades:
            total_fees += trade.fee

            if trade.side == TradeSide.BUY:
                buy_lots.append([trade.quantity, trade.price])
            elif trade.side == TradeSide.SELL:
                remaining_sell = trade.quantity
                sell_price = trade.price

                while remaining_sell > 0 and buy_lots:
                    lot = buy_lots[0]
                    lot_qty, lot_price = lot

                    matched = min(remaining_sell, lot_qty)
                    cost = matched * lot_price
                    proceeds = matched * sell_price

                    total_cost_basis += cost
                    total_proceeds += proceeds
                    total_buy_qty_sold += matched
                    total_buy_cost_sold += cost
                    total_sell_qty += matched
                    total_sell_proceeds += proceeds

                    lot[0] -= matched
                    remaining_sell -= matched

                    if lot[0] <= 1e-12:  # lot fully consumed
                        buy_lots.popleft()

                    close_count += 1

                if remaining_sell > 1e-12:
                    logger.debug(
                        f"{symbol}: sell of {trade.quantity} exceeded known buys "
                        f"by {remaining_sell:.8f} (transfer-in or pre-history)"
                    )

        realized = total_proceeds - total_cost_basis
        pnl_pct = (realized / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        avg_buy = (total_buy_cost_sold / total_buy_qty_sold) if total_buy_qty_sold > 0 else 0.0
        avg_sell = (total_sell_proceeds / total_sell_qty) if total_sell_qty > 0 else 0.0

        if close_count > 0 or total_fees > 0:
            timestamps = [t.timestamp for t in sorted_trades]
            results[symbol] = RealizedPnL(
                symbol=symbol,
                total_realized_pnl=round(realized, 2),
                total_cost_basis=round(total_cost_basis, 2),
                total_proceeds=round(total_proceeds, 2),
                total_fees=round(total_fees, 2),
                trade_count=close_count,
                avg_buy_price=round(avg_buy, 2),
                avg_sell_price=round(avg_sell, 2),
                pnl_pct=round(pnl_pct, 2),
                first_trade=timestamps[0] if timestamps else None,
                last_trade=timestamps[-1] if timestamps else None,
            )

    return results


def compute_current_cost_basis(trades: list[Trade]) -> dict[str, float]:
    """Compute average cost per unit for currently-held positions.

    After FIFO matching of sells against buys, the remaining buy lots
    represent the cost basis of current holdings.

    Returns:
        {symbol: avg_cost_per_unit} for symbols with remaining buy lots.
    """
    by_symbol: dict[str, list[Trade]] = {}
    for t in trades:
        if t.status != TradeStatus.FILLED:
            continue
        by_symbol.setdefault(t.symbol, []).append(t)

    cost_basis: dict[str, float] = {}

    for symbol, symbol_trades in by_symbol.items():
        sorted_trades = sorted(symbol_trades, key=lambda t: t.timestamp)

        buy_lots: deque[list[float]] = deque()

        for trade in sorted_trades:
            if trade.side == TradeSide.BUY:
                buy_lots.append([trade.quantity, trade.price])
            elif trade.side == TradeSide.SELL:
                remaining_sell = trade.quantity
                while remaining_sell > 0 and buy_lots:
                    lot = buy_lots[0]
                    matched = min(remaining_sell, lot[0])
                    lot[0] -= matched
                    remaining_sell -= matched
                    if lot[0] <= 1e-12:
                        buy_lots.popleft()

        # Remaining lots = current holdings cost basis
        total_qty = sum(lot[0] for lot in buy_lots)
        total_cost = sum(lot[0] * lot[1] for lot in buy_lots)

        if total_qty > 1e-12:
            cost_basis[symbol] = round(total_cost / total_qty, 6)

    return cost_basis
