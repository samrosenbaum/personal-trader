"""Derivatives strategy advisor.

Provides specific recommendations for:
- Futures hedging (short to protect long spot positions)
- Put buying (crash protection / insurance)
- Put selling (income generation in fearful markets)
- Basis trades and funding rate arbitrage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from personal_trader.config import RiskProfile
from personal_trader.strategy.risk import RISK_PARAMS

logger = logging.getLogger(__name__)


class DerivativeType(str, Enum):
    FUTURES_SHORT = "futures_short"
    FUTURES_LONG = "futures_long"
    BUY_PUT = "buy_put"
    SELL_PUT = "sell_put"
    BUY_CALL = "buy_call"
    SELL_CALL = "sell_call"
    BASIS_TRADE = "basis_trade"
    FUNDING_ARB = "funding_arbitrage"


@dataclass
class DerivativeStrategy:
    strategy_type: DerivativeType
    symbol: str
    direction: str  # "long" or "short"
    rationale: str
    sizing: str  # description of recommended size
    entry_trigger: str
    exit_trigger: str
    max_loss_pct: float  # max loss as % of portfolio
    expected_pnl: str
    risk_reward: str
    notes: list[str] = field(default_factory=list)


@dataclass
class DerivativesReport:
    strategies: list[DerivativeStrategy] = field(default_factory=list)
    overall_hedge_ratio: float = 0.0  # recommended % of portfolio to hedge
    market_conditions: str = ""

    @property
    def has_recommendations(self) -> bool:
        return len(self.strategies) > 0


class DerivativesAdvisor:
    """Recommends derivative strategies based on market analysis."""

    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE) -> None:
        self.risk_profile = risk_profile
        self.params = RISK_PARAMS[risk_profile]

    def analyze(
        self,
        symbol: str,
        spot_price: float,
        portfolio_allocation_pct: float,
        indicators: pd.DataFrame | None = None,
        sentiment_score: int = 50,
        funding_rate: float | None = None,
        volatility_30d: float | None = None,
    ) -> DerivativesReport:
        """Generate derivative strategy recommendations.

        Args:
            symbol: Trading pair
            spot_price: Current spot price
            portfolio_allocation_pct: How much of portfolio is in this asset
            indicators: DataFrame with technical indicators
            sentiment_score: 0-100 fear/greed score
            funding_rate: Current perpetual funding rate
            volatility_30d: 30-day annualized volatility %
        """
        report = DerivativesReport()
        report.market_conditions = self._assess_conditions(
            sentiment_score, volatility_30d, indicators
        )

        # Determine hedge ratio
        report.overall_hedge_ratio = self._recommended_hedge_ratio(
            sentiment_score, volatility_30d, portfolio_allocation_pct
        )

        # Futures hedging
        if self._should_hedge_futures(sentiment_score, volatility_30d, indicators):
            report.strategies.append(self._futures_hedge_strategy(
                symbol, spot_price, portfolio_allocation_pct, report.overall_hedge_ratio,
                volatility_30d,
            ))

        # Put protection
        if self._should_buy_puts(sentiment_score, volatility_30d, indicators):
            report.strategies.append(self._buy_put_strategy(
                symbol, spot_price, volatility_30d, sentiment_score,
            ))

        # Put selling for income
        if self._should_sell_puts(sentiment_score, volatility_30d, indicators):
            report.strategies.append(self._sell_put_strategy(
                symbol, spot_price, volatility_30d, sentiment_score,
            ))

        # Funding rate arbitrage
        if funding_rate is not None and abs(funding_rate) > 0.0005:
            report.strategies.append(self._funding_arb_strategy(
                symbol, spot_price, funding_rate,
            ))

        return report

    def _assess_conditions(
        self,
        sentiment: int,
        vol: float | None,
        indicators: pd.DataFrame | None,
    ) -> str:
        conditions = []
        if sentiment <= 20:
            conditions.append("Extreme fear - high premiums, contrarian bullish")
        elif sentiment <= 40:
            conditions.append("Fear - elevated premiums")
        elif sentiment >= 80:
            conditions.append("Extreme greed - complacent market, crash risk elevated")
        elif sentiment >= 60:
            conditions.append("Greed - consider protection")

        if vol:
            if vol > 100:
                conditions.append(f"Very high volatility ({vol:.0f}%)")
            elif vol > 60:
                conditions.append(f"Elevated volatility ({vol:.0f}%)")
            elif vol < 30:
                conditions.append(f"Low volatility ({vol:.0f}%) - squeeze potential")

        return "; ".join(conditions) if conditions else "Normal market conditions"

    def _recommended_hedge_ratio(
        self, sentiment: int, vol: float | None, alloc_pct: float
    ) -> float:
        """What % of the position should be hedged."""
        base = 0.0

        # Sentiment-driven
        if sentiment >= 80:
            base = 0.5
        elif sentiment >= 65:
            base = 0.3
        elif sentiment <= 20:
            base = 0.0  # fear = don't hedge, accumulate
        elif sentiment <= 35:
            base = 0.1

        # Volatility adjustment
        if vol and vol > 100:
            base += 0.2
        elif vol and vol > 60:
            base += 0.1

        # Concentration adjustment
        if alloc_pct > 30:
            base += 0.15
        elif alloc_pct > 20:
            base += 0.05

        # Profile cap
        max_hedge = {
            RiskProfile.CONSERVATIVE: 0.7,
            RiskProfile.MODERATE: 0.5,
            RiskProfile.AGGRESSIVE: 0.3,
        }

        return min(base, max_hedge[self.risk_profile])

    def _should_hedge_futures(
        self, sentiment: int, vol: float | None, indicators: pd.DataFrame | None
    ) -> bool:
        if sentiment >= 65:
            return True
        if vol and vol > 80:
            return True
        if indicators is not None and not indicators.empty:
            last = indicators.iloc[-1]
            if last.get("rsi_14") and last["rsi_14"] > 70:
                return True
            if last.get("macd_hist") and last["macd_hist"] < 0:
                if last.get("sma_50") and last.get("sma_200"):
                    if last["sma_50"] < last["sma_200"]:
                        return True
        return False

    def _should_buy_puts(
        self, sentiment: int, vol: float | None, indicators: pd.DataFrame | None
    ) -> bool:
        # Buy puts when greed + bearish divergence
        if sentiment >= 60 and vol and vol < 50:
            return True  # low vol + greed = cheap insurance
        if indicators is not None and not indicators.empty:
            last = indicators.iloc[-1]
            if last.get("rsi_14") and last["rsi_14"] > 75:
                return True
        return False

    def _should_sell_puts(
        self, sentiment: int, vol: float | None, indicators: pd.DataFrame | None
    ) -> bool:
        # Sell puts in fear with bullish technicals
        if sentiment <= 30:
            if indicators is not None and not indicators.empty:
                last = indicators.iloc[-1]
                if last.get("rsi_14") and last["rsi_14"] < 40:
                    return True
                if last.get("close") and last.get("sma_200"):
                    if last["close"] > last["sma_200"]:
                        return True
        return False

    def _futures_hedge_strategy(
        self,
        symbol: str,
        price: float,
        alloc_pct: float,
        hedge_ratio: float,
        vol: float | None,
    ) -> DerivativeStrategy:
        hedge_pct = hedge_ratio * 100
        sl_distance = price * 0.05 if not vol else price * (vol / 100 / np.sqrt(365)) * 2

        return DerivativeStrategy(
            strategy_type=DerivativeType.FUTURES_SHORT,
            symbol=symbol,
            direction="short",
            rationale=f"Hedge {hedge_pct:.0f}% of {symbol} spot position against downside",
            sizing=f"Short futures equal to {hedge_pct:.0f}% of your {symbol} spot holdings "
                   f"(~{alloc_pct * hedge_ratio:.1f}% of portfolio value)",
            entry_trigger=f"Enter at market or on break below nearest support",
            exit_trigger=f"Close hedge when momentum turns bullish or at target profit",
            max_loss_pct=self.params["stop_loss_pct"] * hedge_ratio,
            expected_pnl=f"Offsets {hedge_pct:.0f}% of any spot loss; "
                         f"costs {hedge_pct:.0f}% of any spot gain",
            risk_reward=f"1:1 hedge ratio (protective, not speculative)",
            notes=[
                "Use 1x leverage only for hedging purposes",
                f"Set stop loss on the short at +{sl_distance:.2f} above entry",
                "Close the hedge gradually as bearish signals diminish",
                "Funding rates may cost or earn while position is open",
            ],
        )

    def _buy_put_strategy(
        self,
        symbol: str,
        price: float,
        vol: float | None,
        sentiment: int,
    ) -> DerivativeStrategy:
        otm_distance = 0.15 if vol and vol > 60 else 0.10
        strike = price * (1 - otm_distance)

        return DerivativeStrategy(
            strategy_type=DerivativeType.BUY_PUT,
            symbol=symbol,
            direction="long put",
            rationale="Portfolio insurance against sharp decline",
            sizing=f"Buy puts covering 20-30% of {symbol} spot position",
            entry_trigger=f"Buy {otm_distance*100:.0f}% OTM puts (strike ~${strike:.0f}) "
                          f"expiring 30-60 days out",
            exit_trigger="Sell puts on sharp decline for profit, or let expire if market rises",
            max_loss_pct=2.0,  # premium paid
            expected_pnl=f"Max loss = premium paid (~1-3% of position). "
                         f"Gains 3-10x on >15% crash",
            risk_reward="Asymmetric: small cost, large potential payoff on crash",
            notes=[
                f"Current sentiment ({sentiment}/100) suggests elevated crash risk"
                if sentiment > 60
                else f"Current sentiment ({sentiment}/100) - moderate risk",
                "Look for options exchanges: Deribit, OKX, Binance Options",
                "Consider rolling puts if they approach expiry with value",
                f"Target strike around ${strike:.0f} ({otm_distance*100:.0f}% below spot)",
            ],
        )

    def _sell_put_strategy(
        self,
        symbol: str,
        price: float,
        vol: float | None,
        sentiment: int,
    ) -> DerivativeStrategy:
        otm_distance = 0.20 if vol and vol > 60 else 0.15
        strike = price * (1 - otm_distance)

        return DerivativeStrategy(
            strategy_type=DerivativeType.SELL_PUT,
            symbol=symbol,
            direction="short put",
            rationale="Collect premium income during fearful market. "
                      "If assigned, acquire asset at a discount",
            sizing=f"Sell puts covering no more than {self.params['position_size_pct']}% "
                   f"of portfolio notional",
            entry_trigger=f"Sell {otm_distance*100:.0f}% OTM puts (strike ~${strike:.0f}) "
                          f"expiring 14-30 days out",
            exit_trigger="Buy back at 50% profit or let expire worthless",
            max_loss_pct=self.params["stop_loss_pct"],
            expected_pnl=f"Collect premium (elevated due to {sentiment}/100 fear). "
                         f"Risk: must buy at ${strike:.0f} if price drops below",
            risk_reward="Limited gain (premium) vs potential assignment at strike",
            notes=[
                f"Fear index at {sentiment}/100 inflates premiums - good for selling",
                "Only sell puts on assets you want to own at the strike price",
                "Keep cash/USDC collateral reserved for potential assignment",
                f"Strike ${strike:.0f} represents a {otm_distance*100:.0f}% discount to current price",
            ],
        )

    def _funding_arb_strategy(
        self,
        symbol: str,
        price: float,
        funding_rate: float,
    ) -> DerivativeStrategy:
        direction = "short" if funding_rate > 0 else "long"
        annual_rate = funding_rate * 3 * 365 * 100  # 3x daily funding, annualized

        return DerivativeStrategy(
            strategy_type=DerivativeType.FUNDING_ARB,
            symbol=symbol,
            direction=f"spot {'long' if direction == 'short' else 'short'} + perp {direction}",
            rationale=f"Capture funding rate ({funding_rate*100:.4f}% per 8h, "
                      f"~{annual_rate:.1f}% annualized) with delta-neutral position",
            sizing="Equal size spot and perpetual positions (delta neutral)",
            entry_trigger=f"Enter when funding rate is consistently {'positive' if funding_rate > 0 else 'negative'}",
            exit_trigger="Exit when funding rate normalizes or flips",
            max_loss_pct=1.0,  # slippage and fees mainly
            expected_pnl=f"~{annual_rate:.1f}% annualized from funding payments",
            risk_reward="Low risk, steady income while funding persists",
            notes=[
                f"Current funding: {funding_rate*100:.4f}% per 8h",
                f"Annualized: ~{annual_rate:.1f}%",
                "Hold spot long + short perp (or vice versa) for delta neutrality",
                "Monitor funding rate changes - exit if it drops below 0.01%",
            ],
        )
