"""Multi-exchange connection manager using ccxt."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from personal_trader.config import Settings
from personal_trader.exchanges.trades import (
    Trade,
    TradeHistory,
    TradeSide,
    TradeStatus,
    compute_current_cost_basis,
    compute_realized_pnl,
)

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Snapshot of holdings across all exchanges and brokers."""

    balances: dict[str, float] = field(default_factory=dict)
    total_usd: float = 0.0
    exchange_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def allocation(self) -> dict[str, float]:
        """Return percentage allocation per asset."""
        if self.total_usd == 0:
            return {}
        return {asset: (val / self.total_usd) * 100 for asset, val in self.balances.items() if val > 0.01}


class ExchangeManager:
    """Manages connections to crypto exchanges and Robinhood."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exchanges: dict[str, ccxt.Exchange] = {}
        self._robinhood = None  # lazy-loaded
        self._robinhood_portfolio = None  # cached per session
        self._trade_history: TradeHistory | None = None  # cached per session
        self._init_exchanges()

    def _init_exchanges(self) -> None:
        """Initialize exchange connections based on available API keys."""
        exchange_configs = {
            "binance": {
                "apiKey": self.settings.binance_api_key,
                "secret": self.settings.binance_secret,
                "options": {"defaultType": "spot"},
            },
            "coinbase": {
                "apiKey": self.settings.coinbase_api_key,
                "secret": self.settings.coinbase_secret,
                "password": self.settings.coinbase_passphrase,
            },
            "kraken": {
                "apiKey": self.settings.kraken_api_key,
                "secret": self.settings.kraken_secret,
            },
            "bybit": {
                "apiKey": self.settings.bybit_api_key,
                "secret": self.settings.bybit_secret,
                "options": {"defaultType": "spot"},
            },
        }

        for name, config in exchange_configs.items():
            if config.get("apiKey"):
                try:
                    exchange_class = getattr(ccxt, name)
                    exchange = exchange_class(config)
                    exchange.load_markets()
                    self.exchanges[name] = exchange
                    logger.info(f"Connected to {name}")
                except Exception as e:
                    logger.warning(f"Failed to connect to {name}: {e}")

        if not self.exchanges:
            logger.warning("No exchanges connected. Using public data only.")

    def _load_robinhood_csv(self) -> dict[str, float] | None:
        """Load Robinhood holdings from a CSV export.

        Used as a fallback when API credentials are not configured.
        Accepts flexible column names: symbol/asset/currency/coin for the
        asset, and quantity/qty/amount/balance for the amount.
        """
        csv_path = self.settings.robinhood_holdings_csv.strip()
        if not csv_path:
            return None

        path = Path(csv_path).expanduser()
        if not path.exists():
            logger.warning(f"Robinhood CSV not found: {path}")
            return None

        balances: dict[str, float] = {}
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    symbol = (
                        row.get("asset")
                        or row.get("symbol")
                        or row.get("currency")
                        or row.get("coin")
                        or ""
                    ).strip().upper()
                    if not symbol:
                        continue

                    qty_raw = (
                        row.get("quantity")
                        or row.get("qty")
                        or row.get("amount")
                        or row.get("balance")
                        or "0"
                    )
                    try:
                        quantity = float(str(qty_raw).replace(",", "").strip())
                    except ValueError:
                        continue
                    if quantity <= 0:
                        continue

                    usd_value = self._get_usd_value(
                        self.get_public_exchange(), symbol, quantity
                    )
                    if usd_value > 0:
                        balances[symbol] = balances.get(symbol, 0.0) + usd_value

            logger.info(f"Loaded {len(balances)} assets from Robinhood CSV")
        except Exception as e:
            logger.warning(f"Failed to parse Robinhood CSV: {e}")
            return None

        return balances if balances else None

    def _get_robinhood(self):
        """Lazy-load the Robinhood connector."""
        if self._robinhood is None and self.settings.robinhood_username:
            from personal_trader.exchanges.robinhood import RobinhoodConnector

            self._robinhood = RobinhoodConnector(self.settings)
        return self._robinhood

    def get_robinhood_portfolio(self):
        """Fetch Robinhood portfolio (cached within session)."""
        rh = self._get_robinhood()
        if rh is None:
            return None
        if self._robinhood_portfolio is None:
            self._robinhood_portfolio = rh.get_portfolio()
        return self._robinhood_portfolio

    def refresh_robinhood(self):
        """Force refresh of Robinhood portfolio cache."""
        self._robinhood_portfolio = None
        return self.get_robinhood_portfolio()

    def get_portfolio(self) -> Portfolio:
        """Fetch combined portfolio from all connected exchanges and Robinhood."""
        portfolio = Portfolio()

        # --- Crypto exchanges via ccxt ---
        for name, exchange in self.exchanges.items():
            try:
                balance = exchange.fetch_balance()
                exchange_balances: dict[str, float] = {}

                for asset, info in balance.get("total", {}).items():
                    if info and float(info) > 0:
                        amount = float(info)
                        usd_value = self._get_usd_value(exchange, asset, amount)
                        portfolio.balances[asset] = portfolio.balances.get(asset, 0) + usd_value
                        exchange_balances[asset] = usd_value

                portfolio.exchange_breakdown[name] = exchange_balances
            except Exception as e:
                logger.error(f"Error fetching balance from {name}: {e}")

        # --- Robinhood (live API first, CSV fallback) ---
        rh_portfolio = self.get_robinhood_portfolio()
        if rh_portfolio and rh_portfolio.total_equity > 0:
            rh_balances: dict[str, float] = {}
            for h in rh_portfolio.crypto_holdings:
                portfolio.balances[h.symbol] = portfolio.balances.get(h.symbol, 0) + h.market_value
                rh_balances[h.symbol] = rh_balances.get(h.symbol, 0) + h.market_value
            for h in rh_portfolio.stock_holdings:
                portfolio.balances[h.symbol] = portfolio.balances.get(h.symbol, 0) + h.market_value
                rh_balances[h.symbol] = rh_balances.get(h.symbol, 0) + h.market_value
            if rh_portfolio.cash_balance > 0:
                portfolio.balances["USD"] = portfolio.balances.get("USD", 0) + rh_portfolio.cash_balance
                rh_balances["USD"] = rh_portfolio.cash_balance
            portfolio.exchange_breakdown["robinhood"] = rh_balances
        else:
            # Fallback: try CSV import
            csv_balances = self._load_robinhood_csv()
            if csv_balances:
                portfolio.exchange_breakdown["robinhood_csv"] = csv_balances
                for asset, usd_value in csv_balances.items():
                    portfolio.balances[asset] = portfolio.balances.get(asset, 0.0) + usd_value

        portfolio.total_usd = sum(portfolio.balances.values())
        return portfolio

    def _get_usd_value(self, exchange: ccxt.Exchange, asset: str, amount: float) -> float:
        """Convert an asset amount to USD value."""
        if asset in ("USD", "USDT", "USDC", "BUSD", "DAI"):
            return amount
        try:
            for quote in ("USDT", "USD", "USDC"):
                symbol = f"{asset}/{quote}"
                if symbol in exchange.markets:
                    ticker = exchange.fetch_ticker(symbol)
                    return amount * ticker["last"]
        except Exception:
            pass
        return 0.0

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        exchange_name: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data for a symbol.

        Args:
            symbol: Trading pair like 'BTC/USDT'
            timeframe: Candle timeframe ('1m','5m','15m','1h','4h','1d','1w')
            limit: Number of candles to fetch
            exchange_name: Specific exchange to use, or None for first available
        """
        exchange = self._pick_exchange(exchange_name, symbol)
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_order_book(self, symbol: str, exchange_name: str | None = None) -> dict[str, Any]:
        """Fetch current order book."""
        exchange = self._pick_exchange(exchange_name, symbol)
        return exchange.fetch_order_book(symbol)

    def fetch_ticker(self, symbol: str, exchange_name: str | None = None) -> dict[str, Any]:
        """Fetch current ticker data."""
        exchange = self._pick_exchange(exchange_name, symbol)
        return exchange.fetch_ticker(symbol)

    def get_public_exchange(self) -> ccxt.Exchange:
        """Return a public (no auth) exchange for market data."""
        if self.exchanges:
            return next(iter(self.exchanges.values()))
        # Fallback: create a public binance connection
        exchange = ccxt.binance({"options": {"defaultType": "spot"}})
        exchange.load_markets()
        self.exchanges["binance_public"] = exchange
        return exchange

    def _pick_exchange(self, name: str | None, symbol: str) -> ccxt.Exchange:
        """Pick an exchange that supports the given symbol."""
        if name and name in self.exchanges:
            return self.exchanges[name]
        for ex in self.exchanges.values():
            if symbol in ex.markets:
                return ex
        return self.get_public_exchange()

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def fetch_exchange_trades(
        self,
        exchange_name: str | None = None,
        since: datetime | None = None,
    ) -> list[Trade]:
        """Fetch trade history from CCXT exchanges.

        Iterates over held symbols and calls fetch_my_trades for each.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=90)
        since_ms = int(since.timestamp() * 1000)

        all_trades: list[Trade] = []
        exchanges_to_scan = (
            {exchange_name: self.exchanges[exchange_name]}
            if exchange_name and exchange_name in self.exchanges
            else self.exchanges
        )

        for name, exchange in exchanges_to_scan.items():
            if not exchange.has.get("fetchMyTrades"):
                logger.info(f"{name} does not support fetchMyTrades, skipping")
                continue

            # Get held symbols to know what to query
            symbols_to_query: set[str] = set()
            try:
                balance = exchange.fetch_balance()
                for asset, amount in balance.get("total", {}).items():
                    if amount and float(amount) > 0 and asset not in ("USD", "USDT", "USDC", "BUSD", "DAI"):
                        for quote in ("USDT", "USD", "USDC"):
                            pair = f"{asset}/{quote}"
                            if pair in exchange.markets:
                                symbols_to_query.add(pair)
                                break
            except Exception as e:
                logger.warning(f"Could not fetch balance from {name} for trade history: {e}")

            for symbol in symbols_to_query:
                try:
                    trades = self._fetch_trades_paginated(exchange, symbol, since_ms)
                    for raw in trades:
                        all_trades.append(self._normalize_ccxt_trade(raw, name))
                except Exception as e:
                    logger.warning(f"Failed to fetch trades for {symbol} on {name}: {e}")

                # Respect rate limits
                if hasattr(exchange, "rateLimit"):
                    time.sleep(exchange.rateLimit / 1000)

        return all_trades

    @staticmethod
    def _fetch_trades_paginated(
        exchange: ccxt.Exchange,
        symbol: str,
        since_ms: int,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch all trades for a symbol with pagination."""
        all_trades: list[dict] = []
        cursor = since_ms

        while True:
            batch = exchange.fetch_my_trades(symbol, since=cursor, limit=limit)
            if not batch:
                break
            all_trades.extend(batch)
            if len(batch) < limit:
                break
            cursor = batch[-1]["timestamp"] + 1

        return all_trades

    @staticmethod
    def _normalize_ccxt_trade(raw: dict, exchange_name: str) -> Trade:
        """Convert a ccxt trade dict to our unified Trade dataclass."""
        symbol_parts = raw.get("symbol", "UNKNOWN/USD").split("/")
        base_symbol = symbol_parts[0]

        fee_info = raw.get("fee") or {}
        fee_cost = float(fee_info.get("cost", 0) or 0)

        return Trade(
            id=str(raw.get("id", "")),
            symbol=base_symbol,
            side=TradeSide(raw["side"]),
            quantity=float(raw.get("amount", 0)),
            price=float(raw.get("price", 0)),
            total=float(raw.get("cost", 0) or (raw.get("amount", 0) * raw.get("price", 0))),
            fee=fee_cost,
            timestamp=datetime.fromtimestamp(raw["timestamp"] / 1000, tz=timezone.utc),
            source=exchange_name,
            asset_type="crypto",
            status=TradeStatus.FILLED,
            order_id=str(raw.get("order", "")),
            pair=raw.get("symbol", ""),
        )

    def fetch_all_trade_history(
        self,
        since: datetime | None = None,
    ) -> TradeHistory:
        """Fetch trade history from ALL sources (exchanges + Robinhood).

        Returns a unified TradeHistory with realized P&L computed.
        """
        all_trades: list[Trade] = []

        # Robinhood
        rh = self._get_robinhood()
        if rh is not None:
            try:
                rh_trades = rh.get_all_order_history()
                all_trades.extend(rh_trades)
                logger.info(f"Fetched {len(rh_trades)} trades from Robinhood")
            except Exception as e:
                logger.error(f"Failed to fetch Robinhood trade history: {e}")

        # CCXT exchanges
        try:
            exchange_trades = self.fetch_exchange_trades(since=since)
            all_trades.extend(exchange_trades)
            logger.info(f"Fetched {len(exchange_trades)} trades from exchanges")
        except Exception as e:
            logger.error(f"Failed to fetch exchange trade history: {e}")

        # Sort all trades chronologically
        all_trades.sort(key=lambda t: t.timestamp)

        # Compute realized P&L
        realized = compute_realized_pnl(all_trades)
        total_realized = sum(r.total_realized_pnl for r in realized.values())
        total_fees = sum(t.fee for t in all_trades)

        return TradeHistory(
            trades=all_trades,
            realized_pnl=realized,
            total_realized_pnl=round(total_realized, 2),
            total_fees=round(total_fees, 2),
        )

    def get_trade_history(self, since: datetime | None = None) -> TradeHistory:
        """Fetch and cache trade history from all sources."""
        if self._trade_history is None:
            self._trade_history = self.fetch_all_trade_history(since)
        return self._trade_history

    def get_cost_basis(self) -> dict[str, float]:
        """Get current cost basis per symbol from trade history.

        Returns {symbol: avg_cost_per_unit} for currently-held positions.
        """
        history = self.get_trade_history()
        return compute_current_cost_basis(history.trades)
