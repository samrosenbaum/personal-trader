"""Multi-exchange connection manager using ccxt."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import ccxt
import pandas as pd

from personal_trader.config import Settings

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

        # --- Robinhood ---
        rh_portfolio = self.get_robinhood_portfolio()
        if rh_portfolio:
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
