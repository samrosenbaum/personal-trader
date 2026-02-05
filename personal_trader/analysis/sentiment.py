"""Market sentiment analysis.

Fetches and analyses:
- Fear & Greed Index (alternative.me)
- Funding rates (exchange)
- Open interest trends
- Long/short ratios
- Social sentiment proxies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=30&format=json"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"


class SentimentLevel(str, Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class FearGreedData:
    value: int  # 0-100
    label: str
    level: SentimentLevel
    timestamp: datetime
    history: list[tuple[datetime, int]]  # last 30 days


@dataclass
class MarketOverview:
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    eth_dominance: float
    market_cap_change_24h: float
    active_cryptos: int


@dataclass
class FundingSnapshot:
    symbol: str
    funding_rate: float
    next_funding_time: datetime | None
    predicted_rate: float | None


@dataclass
class SentimentReport:
    fear_greed: FearGreedData | None = None
    market_overview: MarketOverview | None = None
    funding_rates: list[FundingSnapshot] | None = None
    trending_coins: list[str] | None = None

    @property
    def overall_sentiment(self) -> SentimentLevel:
        if self.fear_greed:
            return self.fear_greed.level
        return SentimentLevel.NEUTRAL

    @property
    def sentiment_score(self) -> int:
        """0 = extreme fear, 100 = extreme greed."""
        if self.fear_greed:
            return self.fear_greed.value
        return 50


class SentimentAnalyzer:
    """Collects and analyses market sentiment data."""

    def __init__(self, exchange_manager=None) -> None:
        self.exchange_manager = exchange_manager
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "personal-trader/0.1"})

    def get_full_report(self) -> SentimentReport:
        report = SentimentReport()
        report.fear_greed = self._fetch_fear_greed()
        report.market_overview = self._fetch_market_overview()
        report.trending_coins = self._fetch_trending()
        if self.exchange_manager:
            report.funding_rates = self._fetch_funding_rates()
        return report

    def _fetch_fear_greed(self) -> FearGreedData | None:
        try:
            resp = self.session.get(FEAR_GREED_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()["data"]
            current = data[0]
            value = int(current["value"])
            history = [
                (datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc), int(d["value"]))
                for d in data
            ]
            return FearGreedData(
                value=value,
                label=current["value_classification"],
                level=self._classify_fear_greed(value),
                timestamp=datetime.fromtimestamp(int(current["timestamp"]), tz=timezone.utc),
                history=history,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed: {e}")
            return None

    def _fetch_market_overview(self) -> MarketOverview | None:
        try:
            resp = self.session.get(COINGECKO_GLOBAL_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()["data"]
            return MarketOverview(
                total_market_cap=data["total_market_cap"].get("usd", 0),
                total_volume_24h=data["total_volume"].get("usd", 0),
                btc_dominance=data["market_cap_percentage"].get("btc", 0),
                eth_dominance=data["market_cap_percentage"].get("eth", 0),
                market_cap_change_24h=data.get("market_cap_change_percentage_24h_usd", 0),
                active_cryptos=data.get("active_cryptocurrencies", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch market overview: {e}")
            return None

    def _fetch_trending(self) -> list[str] | None:
        try:
            resp = self.session.get(COINGECKO_TRENDING_URL, timeout=10)
            resp.raise_for_status()
            coins = resp.json().get("coins", [])
            return [c["item"]["symbol"].upper() for c in coins[:10]]
        except Exception as e:
            logger.warning(f"Failed to fetch trending: {e}")
            return None

    def _fetch_funding_rates(self) -> list[FundingSnapshot]:
        """Fetch funding rates from connected derivatives exchanges."""
        results = []
        symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

        for name, exchange in self.exchange_manager.exchanges.items():
            if not hasattr(exchange, "fetch_funding_rate"):
                continue
            for symbol in symbols:
                try:
                    funding = exchange.fetch_funding_rate(symbol)
                    results.append(FundingSnapshot(
                        symbol=symbol,
                        funding_rate=funding.get("fundingRate", 0),
                        next_funding_time=None,
                        predicted_rate=funding.get("fundingRatePredicted"),
                    ))
                except Exception:
                    pass
        return results

    @staticmethod
    def _classify_fear_greed(value: int) -> SentimentLevel:
        if value <= 20:
            return SentimentLevel.EXTREME_FEAR
        if value <= 40:
            return SentimentLevel.FEAR
        if value <= 60:
            return SentimentLevel.NEUTRAL
        if value <= 80:
            return SentimentLevel.GREED
        return SentimentLevel.EXTREME_GREED
