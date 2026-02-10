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
from dataclasses import dataclass, field
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


class DirectionLabel(str, Enum):
    STRONGLY_BEARISH = "strongly_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    STRONGLY_BULLISH = "strongly_bullish"


@dataclass
class DirectionComponent:
    """A single component contributing to the market direction index."""
    name: str
    raw_value: float | None
    score: float
    weight: float
    description: str


@dataclass
class MarketDirectionIndex:
    """Aggregated market direction index from -100 (bearish) to +100 (bullish)."""
    score: float
    label: DirectionLabel
    components: list[DirectionComponent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def label_display(self) -> str:
        return self.label.value.replace("_", " ").upper()


@dataclass
class SentimentReport:
    fear_greed: FearGreedData | None = None
    market_overview: MarketOverview | None = None
    funding_rates: list[FundingSnapshot] | None = None
    trending_coins: list[str] | None = None
    direction_index: MarketDirectionIndex | None = None

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

    def compute_market_direction_index(
        self,
        report: SentimentReport,
        btc_technical_score: float | None = None,
        eth_technical_score: float | None = None,
    ) -> MarketDirectionIndex:
        """Compute an aggregated market direction index from -100 to +100.

        Combines five components:
        1. Fear & Greed (contrarian-inverted) — 20%
        2. BTC + ETH technical composite (bellwether) — 35%
        3. Market cap 24h change direction — 15%
        4. Funding rate pressure — 15%
        5. BTC dominance trend — 15%
        """
        components: list[DirectionComponent] = []

        # 1. Fear & Greed (contrarian) — weight 0.20
        fg_score = 0.0
        fg_raw = None
        fg_desc = "Not available"
        if report.fear_greed:
            fg_raw = float(report.fear_greed.value)
            fg_score = max(-100, min(100, (50 - fg_raw) * 2))
            if fg_raw <= 25:
                fg_desc = f"Extreme fear ({fg_raw:.0f}/100) = contrarian bullish"
            elif fg_raw >= 75:
                fg_desc = f"Extreme greed ({fg_raw:.0f}/100) = contrarian bearish"
            else:
                fg_desc = f"Fear & Greed at {fg_raw:.0f}/100 = neutral signal"
        components.append(DirectionComponent(
            name="Fear & Greed (contrarian)",
            raw_value=fg_raw, score=fg_score, weight=0.20, description=fg_desc,
        ))

        # 2. BTC + ETH technical bellwether — weight 0.35
        bw_score = 0.0
        bw_raw = None
        bw_desc = "Not available (no BTC/ETH technical data)"
        if btc_technical_score is not None or eth_technical_score is not None:
            btc_s = btc_technical_score or 0
            eth_s = eth_technical_score or 0
            if btc_technical_score is not None and eth_technical_score is not None:
                bw_score = btc_s * 0.6 + eth_s * 0.4
            elif btc_technical_score is not None:
                bw_score = btc_s
            else:
                bw_score = eth_s
            bw_raw = bw_score
            direction = "bullish" if bw_score > 20 else "bearish" if bw_score < -20 else "neutral"
            bw_desc = f"BTC+ETH technicals {direction} (score: {bw_score:+.0f})"
        components.append(DirectionComponent(
            name="BTC + ETH Bellwether",
            raw_value=bw_raw, score=bw_score, weight=0.35, description=bw_desc,
        ))

        # 3. Market cap 24h change — weight 0.15
        mcap_score = 0.0
        mcap_raw = None
        mcap_desc = "Not available"
        if report.market_overview:
            mcap_raw = report.market_overview.market_cap_change_24h
            mcap_score = max(-100, min(100, mcap_raw * 20))
            direction = "up" if mcap_raw > 0 else "down"
            mcap_desc = f"Market cap 24h: {mcap_raw:+.2f}% ({direction})"
        components.append(DirectionComponent(
            name="Market Cap 24h Change",
            raw_value=mcap_raw, score=mcap_score, weight=0.15, description=mcap_desc,
        ))

        # 4. Funding rate pressure — weight 0.15
        fund_score = 0.0
        fund_raw = None
        fund_desc = "Not available (no funding data)"
        if report.funding_rates:
            avg_rate = sum(f.funding_rate for f in report.funding_rates) / len(report.funding_rates)
            fund_raw = avg_rate
            fund_score = max(-100, min(100, -avg_rate * 100000))
            if avg_rate > 0.0005:
                fund_desc = f"High positive funding ({avg_rate*100:.4f}%) = overleveraged longs"
            elif avg_rate < -0.0005:
                fund_desc = f"Negative funding ({avg_rate*100:.4f}%) = overleveraged shorts"
            else:
                fund_desc = f"Funding neutral ({avg_rate*100:.4f}%)"
        components.append(DirectionComponent(
            name="Funding Rate Pressure",
            raw_value=fund_raw, score=fund_score, weight=0.15, description=fund_desc,
        ))

        # 5. BTC dominance trend — weight 0.15
        dom_score = 0.0
        dom_raw = None
        dom_desc = "Not available"
        if report.market_overview:
            dom_raw = report.market_overview.btc_dominance
            dom_score = max(-100, min(100, (50 - dom_raw) * 4))
            if dom_raw > 55:
                dom_desc = f"BTC dominance high ({dom_raw:.1f}%) = risk-off / bearish alts"
            elif dom_raw < 45:
                dom_desc = f"BTC dominance low ({dom_raw:.1f}%) = alt season / bullish"
            else:
                dom_desc = f"BTC dominance neutral ({dom_raw:.1f}%)"
        components.append(DirectionComponent(
            name="BTC Dominance Trend",
            raw_value=dom_raw, score=dom_score, weight=0.15, description=dom_desc,
        ))

        # Weighted composite
        total = max(-100, min(100, sum(c.score * c.weight for c in components)))

        if total <= -60:
            label = DirectionLabel.STRONGLY_BEARISH
        elif total <= -20:
            label = DirectionLabel.BEARISH
        elif total < 20:
            label = DirectionLabel.NEUTRAL
        elif total < 60:
            label = DirectionLabel.BULLISH
        else:
            label = DirectionLabel.STRONGLY_BULLISH

        return MarketDirectionIndex(
            score=round(total, 1),
            label=label,
            components=components,
        )

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
