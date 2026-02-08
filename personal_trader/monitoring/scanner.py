"""Continuous market scanner.

Runs on a configurable interval, scans all portfolio assets (and watchlist),
generates signals, and dispatches alerts when action is needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from personal_trader.analysis.derivatives_market import fetch_futures_basis, fetch_vol_surface
from personal_trader.analysis.microstructure import (
    anchored_vwap,
    analyze_order_book,
    volume_profile,
)
from personal_trader.analysis.patterns import PatternDetector
from personal_trader.analysis.sentiment import SentimentAnalyzer
from personal_trader.analysis.technical import compute_multi_timeframe
from personal_trader.config import Settings
from personal_trader.exchanges.manager import ExchangeManager
from personal_trader.monitoring.alerts import AlertDispatcher
from personal_trader.strategy.derivatives import DerivativesAdvisor
from personal_trader.strategy.risk import RiskManager
from personal_trader.strategy.signals import Action, Signal, SignalGenerator, Urgency

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    symbol: str
    timestamp: datetime
    signals: list[Signal] = field(default_factory=list)
    price: float = 0.0


@dataclass
class ScanCycle:
    timestamp: datetime
    results: list[ScanResult] = field(default_factory=list)
    portfolio_value: float = 0.0
    actionable_count: int = 0
    errors: list[str] = field(default_factory=list)


class MarketScanner:
    """Continuously scans markets and generates trading signals."""

    DEFAULT_WATCHLIST = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT",
    ]

    def __init__(
        self,
        settings: Settings,
        exchange_manager: ExchangeManager,
        alert_dispatcher: AlertDispatcher | None = None,
    ) -> None:
        self.settings = settings
        self.exchange_manager = exchange_manager
        self.signal_gen = SignalGenerator(settings.risk_profile)
        self.risk_mgr = RiskManager(settings.risk_profile)
        self.deriv_advisor = DerivativesAdvisor(settings.risk_profile)
        self.sentiment_analyzer = SentimentAnalyzer(exchange_manager)
        self.alert_dispatcher = alert_dispatcher
        self.watchlist = list(self.DEFAULT_WATCHLIST)
        self._last_scan: ScanCycle | None = None
        self._vol_surface_cache: dict[str, tuple[object, datetime]] = {}
        self._futures_basis_cache: dict[str, tuple[object, datetime]] = {}

    def add_to_watchlist(self, symbol: str) -> None:
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)

    def remove_from_watchlist(self, symbol: str) -> None:
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)

    def scan_once(self) -> ScanCycle:
        """Run one full scan cycle across all watched symbols."""
        cycle = ScanCycle(
            timestamp=datetime.now(timezone.utc),
        )

        # Fetch sentiment once per cycle
        try:
            sentiment = self.sentiment_analyzer.get_full_report()
        except Exception as e:
            sentiment = None
            cycle.errors.append(f"Sentiment fetch failed: {e}")

        # Fetch portfolio
        try:
            portfolio = self.exchange_manager.get_portfolio()
            cycle.portfolio_value = portfolio.total_usd
        except Exception as e:
            cycle.errors.append(f"Portfolio fetch failed: {e}")

        # Build scan list: portfolio assets + watchlist
        symbols_to_scan = set(self.watchlist)
        if portfolio and portfolio.allocation:
            for asset in portfolio.allocation:
                pair = f"{asset}/USDT"
                symbols_to_scan.add(pair)

        for symbol in symbols_to_scan:
            try:
                result = self._scan_symbol(symbol, sentiment, cycle.portfolio_value)
                cycle.results.append(result)

                # Count actionable signals
                for sig in result.signals:
                    if sig.action not in (Action.HOLD,) and sig.urgency in (
                        Urgency.HIGH,
                        Urgency.IMMEDIATE,
                    ):
                        cycle.actionable_count += 1

            except Exception as e:
                cycle.errors.append(f"{symbol}: {e}")
                logger.error(f"Error scanning {symbol}: {e}")

        # Dispatch alerts for urgent signals
        if self.alert_dispatcher:
            urgent = [
                sig
                for r in cycle.results
                for sig in r.signals
                if sig.urgency in (Urgency.HIGH, Urgency.IMMEDIATE)
            ]
            for sig in urgent:
                self.alert_dispatcher.send_signal_alert(sig)

        self._last_scan = cycle
        return cycle

    def _scan_symbol(
        self,
        symbol: str,
        sentiment=None,
        portfolio_value: float | None = None,
    ) -> ScanResult:
        """Scan a single symbol across multiple timeframes."""
        result = ScanResult(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )

        # Fetch multi-timeframe data
        tf_data = compute_multi_timeframe(
            fetch_fn=self.exchange_manager.fetch_ohlcv,
            symbol=symbol,
            timeframes=["15m", "1h", "4h", "1d"],
        )

        if not tf_data:
            return result

        # Get current price
        for tf in ("1h", "4h", "1d", "15m"):
            if tf in tf_data and not tf_data[tf].empty:
                result.price = float(tf_data[tf].iloc[-1]["close"])
                break

        # Detect patterns on daily timeframe
        patterns = None
        if "1d" in tf_data:
            try:
                detector = PatternDetector(tf_data["1d"])
                patterns = detector.detect_all()
            except Exception:
                pass

        # Microstructure
        micro = None
        try:
            order_book = self.exchange_manager.fetch_order_book(symbol)
            micro = analyze_order_book(order_book)
        except Exception:
            pass

        # Volume profile + anchored VWAP
        profile = volume_profile(tf_data.get("1d")) if "1d" in tf_data else None
        anchored = anchored_vwap(tf_data.get("1d")) if "1d" in tf_data else None

        # Derivatives data (only for major assets to avoid API spam)
        base = symbol.split("/")[0].upper()
        vol_surf = None
        fut_basis = None
        if base in ("BTC", "ETH"):
            vol_surf = self._get_cached_vol_surface(base)
            fut_basis = self._get_cached_futures_basis(base)

        # Generate signals
        signals = self.signal_gen.generate(
            symbol=symbol,
            indicators=tf_data,
            patterns=patterns,
            sentiment=sentiment,
            microstructure=micro,
            volume_profile=profile,
            anchored_vwap=anchored,
            vol_surface=vol_surf,
            futures_basis=fut_basis,
            portfolio_value=portfolio_value,
        )
        result.signals = signals

        return result

    def run_continuous(self, interval: int | None = None) -> None:
        """Run the scanner in a continuous loop.

        Args:
            interval: Seconds between scans (default from settings)
        """
        interval = interval or self.settings.scan_interval
        logger.info(f"Starting continuous scanner (interval: {interval}s)")
        logger.info(f"Watching {len(self.watchlist)} symbols")

        while True:
            try:
                logger.info("Starting scan cycle...")
                cycle = self.scan_once()
                logger.info(
                    f"Scan complete: {len(cycle.results)} symbols, "
                    f"{cycle.actionable_count} actionable signals"
                )
                if cycle.errors:
                    logger.warning(f"Scan errors: {len(cycle.errors)}")
            except Exception as e:
                logger.error(f"Scan cycle failed: {e}")

            time.sleep(interval)

    @property
    def last_scan(self) -> ScanCycle | None:
        return self._last_scan

    def _get_cached_vol_surface(self, asset: str):
        now = datetime.now(timezone.utc)
        cached = self._vol_surface_cache.get(asset)
        if cached and (now - cached[1]).seconds < 3600:
            return cached[0]
        surface = fetch_vol_surface(asset)
        if surface:
            self._vol_surface_cache[asset] = (surface, now)
        return surface

    def _get_cached_futures_basis(self, asset: str):
        now = datetime.now(timezone.utc)
        cached = self._futures_basis_cache.get(asset)
        if cached and (now - cached[1]).seconds < 600:
            return cached[0]
        basis = fetch_futures_basis(asset)
        if basis:
            self._futures_basis_cache[asset] = (basis, now)
        return basis
