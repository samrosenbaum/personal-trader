"""Robinhood portfolio connector.

Uses robin_stocks to:
- Authenticate (username/password + optional TOTP 2FA)
- Fetch crypto holdings with current market values
- Fetch stock/ETF holdings with current market values
- Fetch options positions
- Fetch crypto and stock order history for P&L tracking
- Provide a unified view alongside exchange portfolios

This is READ-ONLY - no trades are placed through this connector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from personal_trader.config import Settings
from personal_trader.exchanges.trades import Trade, TradeSide, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class RobinhoodHolding:
    """A single holding on Robinhood."""

    symbol: str
    name: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    asset_type: str  # "crypto", "stock", "etf", "option"


@dataclass
class RobinhoodOptionPosition:
    """An open options position."""

    symbol: str  # underlying
    option_type: str  # "call" or "put"
    strike: float
    expiration: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    direction: str  # "long" or "short"


@dataclass
class RobinhoodPortfolio:
    """Full Robinhood account snapshot."""

    crypto_holdings: list[RobinhoodHolding] = field(default_factory=list)
    stock_holdings: list[RobinhoodHolding] = field(default_factory=list)
    option_positions: list[RobinhoodOptionPosition] = field(default_factory=list)
    cash_balance: float = 0.0
    crypto_buying_power: float = 0.0
    total_equity: float = 0.0

    @property
    def crypto_value(self) -> float:
        return sum(h.market_value for h in self.crypto_holdings)

    @property
    def stock_value(self) -> float:
        return sum(h.market_value for h in self.stock_holdings)

    @property
    def option_value(self) -> float:
        return sum(o.market_value for o in self.option_positions)

    @property
    def all_holdings_by_symbol(self) -> dict[str, float]:
        """Merge all holdings into {symbol: usd_value} for portfolio integration."""
        combined: dict[str, float] = {}
        for h in self.crypto_holdings:
            combined[h.symbol] = combined.get(h.symbol, 0) + h.market_value
        for h in self.stock_holdings:
            combined[h.symbol] = combined.get(h.symbol, 0) + h.market_value
        if self.cash_balance > 0:
            combined["USD"] = combined.get("USD", 0) + self.cash_balance
        return combined


class RobinhoodConnector:
    """Read-only connector to Robinhood accounts via robin_stocks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._logged_in = False

    def login(self) -> bool:
        """Authenticate with Robinhood.

        Supports TOTP-based 2FA if robinhood_totp_secret is set.
        """
        if not self.settings.robinhood_username or not self.settings.robinhood_password:
            logger.info("Robinhood credentials not configured, skipping")
            return False

        try:
            import robin_stocks.robinhood as rh

            login_kwargs = {
                "username": self.settings.robinhood_username,
                "password": self.settings.robinhood_password,
                "store_session": True,
            }

            # If TOTP secret is configured, generate the code automatically
            if self.settings.robinhood_totp_secret:
                import pyotp

                totp = pyotp.TOTP(self.settings.robinhood_totp_secret)
                login_kwargs["mfa_code"] = totp.now()

            rh.login(**login_kwargs)
            self._logged_in = True
            logger.info("Logged in to Robinhood")
            return True

        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")
            self._logged_in = False
            return False

    def get_portfolio(self) -> RobinhoodPortfolio:
        """Fetch full portfolio from Robinhood."""
        if not self._logged_in:
            if not self.login():
                return RobinhoodPortfolio()

        portfolio = RobinhoodPortfolio()

        portfolio.crypto_holdings = self._fetch_crypto()
        portfolio.stock_holdings = self._fetch_stocks()
        portfolio.option_positions = self._fetch_options()
        portfolio.cash_balance = self._fetch_cash()
        portfolio.crypto_buying_power = self._fetch_crypto_buying_power()
        portfolio.total_equity = (
            portfolio.crypto_value
            + portfolio.stock_value
            + portfolio.option_value
            + portfolio.cash_balance
        )

        return portfolio

    def _fetch_crypto(self) -> list[RobinhoodHolding]:
        """Fetch crypto holdings."""
        holdings = []
        try:
            import robin_stocks.robinhood as rh

            crypto_positions = rh.crypto.get_crypto_positions()
            for pos in crypto_positions:
                quantity = float(pos.get("quantity", 0) or 0)
                if quantity <= 0:
                    continue

                cost_bases = pos.get("cost_bases", [])
                avg_cost = 0.0
                if cost_bases:
                    direct = [cb for cb in cost_bases if cb.get("direct_quantity")]
                    if direct:
                        total_cost = sum(float(cb.get("direct_cost_basis", 0) or 0) for cb in direct)
                        total_qty = sum(float(cb.get("direct_quantity", 0) or 0) for cb in direct)
                        avg_cost = total_cost / total_qty if total_qty > 0 else 0

                # Get currency code from the position
                currency = pos.get("currency", {})
                symbol = currency.get("code", "UNKNOWN")

                # Fetch current price
                try:
                    quote = rh.crypto.get_crypto_quote(symbol)
                    current_price = float(quote.get("mark_price", 0) or 0)
                except Exception:
                    current_price = avg_cost

                market_value = quantity * current_price
                unrealized_pnl = market_value - (quantity * avg_cost) if avg_cost > 0 else 0
                unrealized_pnl_pct = (unrealized_pnl / (quantity * avg_cost) * 100) if avg_cost > 0 else 0

                holdings.append(RobinhoodHolding(
                    symbol=symbol,
                    name=currency.get("name", symbol),
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    asset_type="crypto",
                ))

        except Exception as e:
            logger.error(f"Error fetching Robinhood crypto: {e}")

        return holdings

    def _fetch_stocks(self) -> list[RobinhoodHolding]:
        """Fetch stock and ETF holdings."""
        holdings = []
        try:
            import robin_stocks.robinhood as rh

            stock_positions = rh.account.build_holdings()

            for symbol, data in stock_positions.items():
                quantity = float(data.get("quantity", 0) or 0)
                if quantity <= 0:
                    continue

                avg_cost = float(data.get("average_buy_price", 0) or 0)
                current_price = float(data.get("price", 0) or 0)
                equity = float(data.get("equity", 0) or 0)
                pct_change = float(data.get("percent_change", 0) or 0)
                pnl = float(data.get("equity_change", 0) or 0)
                name = data.get("name", symbol)

                # Determine if ETF or stock based on type
                asset_type = "etf" if data.get("type") == "etp" else "stock"

                holdings.append(RobinhoodHolding(
                    symbol=symbol,
                    name=name,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=equity,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pct_change,
                    asset_type=asset_type,
                ))

        except Exception as e:
            logger.error(f"Error fetching Robinhood stocks: {e}")

        return holdings

    def _fetch_options(self) -> list[RobinhoodOptionPosition]:
        """Fetch open options positions."""
        positions = []
        try:
            import robin_stocks.robinhood as rh

            option_positions = rh.options.get_open_option_positions()

            for pos in option_positions:
                quantity = float(pos.get("quantity", 0) or 0)
                if quantity == 0:
                    continue

                chain_symbol = pos.get("chain_symbol", "???")
                option_type = pos.get("type", "unknown")  # "call" or "put"
                strike = float(pos.get("strike_price", 0) or 0)
                expiration = pos.get("expiration_date", "")
                avg_cost = float(pos.get("average_price", 0) or 0) / 100  # stored in cents
                direction = "long" if pos.get("direction") == "debit" else "short"

                # Get current mark price
                try:
                    option_data = rh.options.get_option_market_data_by_id(pos["option_id"])
                    if option_data and isinstance(option_data, list):
                        option_data = option_data[0]
                    current_price = float(option_data.get("adjusted_mark_price", 0) or 0)
                except Exception:
                    current_price = avg_cost

                market_value = quantity * current_price * 100  # options are 100 shares
                cost_basis = quantity * avg_cost * 100
                pnl = market_value - cost_basis if direction == "long" else cost_basis - market_value

                positions.append(RobinhoodOptionPosition(
                    symbol=chain_symbol,
                    option_type=option_type,
                    strike=strike,
                    expiration=expiration,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=pnl,
                    direction=direction,
                ))

        except Exception as e:
            logger.error(f"Error fetching Robinhood options: {e}")

        return positions

    def _fetch_cash(self) -> float:
        """Fetch available cash balance."""
        try:
            import robin_stocks.robinhood as rh

            profile = rh.profiles.load_account_profile()
            # portfolio_cash is the uninvested cash in the account
            return float(profile.get("portfolio_cash", 0) or 0)
        except Exception as e:
            logger.error(f"Error fetching Robinhood cash: {e}")
            return 0.0

    def _fetch_crypto_buying_power(self) -> float:
        """Fetch crypto-specific buying power."""
        try:
            import robin_stocks.robinhood as rh

            profile = rh.profiles.load_account_profile()
            return float(profile.get("crypto_buying_power", 0) or 0)
        except Exception:
            return 0.0

    def get_crypto_order_history(self) -> list[Trade]:
        """Fetch completed crypto order history from Robinhood."""
        if not self._logged_in:
            if not self.login():
                return []

        trades = []
        try:
            import robin_stocks.robinhood as rh

            orders = rh.crypto.get_crypto_orders()
            if not orders:
                return []

            for order in orders:
                state = order.get("state", "")
                if state not in ("filled",):
                    continue

                side_str = order.get("side", "").lower()
                if side_str not in ("buy", "sell"):
                    continue

                quantity = float(order.get("cumulative_quantity", 0) or 0)
                if quantity <= 0:
                    continue

                avg_price = float(order.get("average_price", 0) or 0)
                if avg_price <= 0:
                    continue

                # Extract symbol from currency_pair_id or type field
                symbol = "UNKNOWN"
                # Try to get from the order's currency code
                currency_code = order.get("currency_code", "")
                if currency_code:
                    symbol = currency_code.upper()
                else:
                    # Fallback: parse from the pair display name if available
                    pair_id = order.get("currency_pair_id", "")
                    if pair_id:
                        symbol = pair_id.split("-")[0].upper() if "-" in pair_id else pair_id

                # Parse timestamp
                created_at = order.get("created_at", "")
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    ts = datetime.now(timezone.utc)

                total = quantity * avg_price
                # Robinhood embeds fees in spread; explicit fee field may not exist
                fee = float(order.get("fees", 0) or 0)

                trades.append(Trade(
                    id=order.get("id", ""),
                    symbol=symbol,
                    side=TradeSide.BUY if side_str == "buy" else TradeSide.SELL,
                    quantity=quantity,
                    price=avg_price,
                    total=total,
                    fee=fee,
                    timestamp=ts,
                    source="robinhood",
                    asset_type="crypto",
                    status=TradeStatus.FILLED,
                    order_id=order.get("id", ""),
                ))

        except Exception as e:
            logger.error(f"Error fetching Robinhood crypto orders: {e}")

        return trades

    def get_stock_order_history(self) -> list[Trade]:
        """Fetch completed stock/ETF order history from Robinhood."""
        if not self._logged_in:
            if not self.login():
                return []

        trades = []
        try:
            import robin_stocks.robinhood as rh

            orders = rh.orders.get_all_stock_orders()
            if not orders:
                return []

            for order in orders:
                state = order.get("state", "")
                if state not in ("filled",):
                    continue

                side_str = order.get("side", "").lower()
                if side_str not in ("buy", "sell"):
                    continue

                quantity = float(order.get("cumulative_quantity", 0) or 0)
                if quantity <= 0:
                    continue

                avg_price = float(order.get("average_price", 0) or 0)
                if avg_price <= 0:
                    continue

                # Get symbol from instrument URL
                symbol = "UNKNOWN"
                instrument_url = order.get("instrument", "")
                if instrument_url:
                    try:
                        instrument_data = rh.stocks.get_instrument_by_url(instrument_url)
                        symbol = instrument_data.get("symbol", "UNKNOWN")
                    except Exception:
                        pass

                created_at = order.get("created_at", "")
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    ts = datetime.now(timezone.utc)

                total = quantity * avg_price
                fee = float(order.get("fees", 0) or 0)

                trades.append(Trade(
                    id=order.get("id", ""),
                    symbol=symbol,
                    side=TradeSide.BUY if side_str == "buy" else TradeSide.SELL,
                    quantity=quantity,
                    price=avg_price,
                    total=total,
                    fee=fee,
                    timestamp=ts,
                    source="robinhood",
                    asset_type="stock",
                    status=TradeStatus.FILLED,
                    order_id=order.get("id", ""),
                ))

        except Exception as e:
            logger.error(f"Error fetching Robinhood stock orders: {e}")

        return trades

    def get_all_order_history(self) -> list[Trade]:
        """Fetch all order history (crypto + stocks), sorted by time."""
        trades = self.get_crypto_order_history()
        trades.extend(self.get_stock_order_history())
        trades.sort(key=lambda t: t.timestamp, reverse=True)
        return trades

    def logout(self) -> None:
        """Log out of Robinhood."""
        if self._logged_in:
            try:
                import robin_stocks.robinhood as rh

                rh.logout()
                self._logged_in = False
            except Exception:
                pass
