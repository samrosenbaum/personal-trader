"""Opportunities engine.

Given what you actually hold, this module generates specific, actionable
money-making ideas based on current chart signals and market conditions.

For each holding it answers: "What can I do with THIS position RIGHT NOW
to profit or protect myself?"

Categories:
  1. Ride the trend      - Hold / add, chart says momentum is in your favor
  2. Take profit         - Chart says overbought, sell some into strength
  3. Rotate to USDC      - Risk-off, convert to stablecoin and sit on sidelines
  4. Hedge it            - Keep the position but open a futures short or buy puts
  5. Sell covered calls   - Earn income on a stock/crypto you hold
  6. Sell cash-secured puts - Use your cash to collect premium
  7. Buy the dip          - Asset is oversold, add to your position
  8. Cut the loss         - Stop loss triggered, cut it before it gets worse
  9. Rebalance            - Position is too concentrated, trim and diversify
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from personal_trader.analysis.sentiment import SentimentReport
from personal_trader.config import RiskProfile
from personal_trader.strategy.risk import RISK_PARAMS, STABLECOINS

logger = logging.getLogger(__name__)


class OpportunityType(str, Enum):
    RIDE_TREND = "ride_the_trend"
    TAKE_PROFIT = "take_profit"
    ROTATE_TO_USDC = "rotate_to_usdc"
    HEDGE_WITH_FUTURES = "hedge_with_futures"
    BUY_PUTS = "buy_puts"
    SELL_COVERED_CALLS = "sell_covered_calls"
    SELL_CASH_SECURED_PUTS = "sell_cash_secured_puts"
    BUY_THE_DIP = "buy_the_dip"
    CUT_LOSS = "cut_loss"
    REBALANCE = "rebalance"
    DCA_ACCUMULATE = "dca_accumulate"
    FUNDING_ARBITRAGE = "funding_arbitrage"


@dataclass
class Opportunity:
    """A specific, actionable recommendation for a held position."""

    opp_type: OpportunityType
    symbol: str
    headline: str  # e.g. "Take 25% profit on BTC - RSI overbought on 4h"
    details: str  # fuller explanation
    action_steps: list[str]  # step-by-step what to do
    potential_gain: str  # estimated upside
    risk: str  # what could go wrong
    confidence: float  # 0-1
    priority: int  # 1=do this now, 2=soon, 3=consider

    # Context about the position
    current_value: float = 0.0
    allocation_pct: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class OpportunitiesReport:
    """All opportunities across the portfolio."""

    opportunities: list[Opportunity] = field(default_factory=list)
    total_portfolio_value: float = 0.0
    cash_available: float = 0.0
    market_regime: str = ""

    @property
    def by_priority(self) -> list[Opportunity]:
        return sorted(self.opportunities, key=lambda o: (o.priority, -o.confidence))

    @property
    def top_opportunities(self) -> list[Opportunity]:
        return self.by_priority[:10]


class OpportunitiesEngine:
    """Analyzes held positions and generates money-making opportunities."""

    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE) -> None:
        self.risk_profile = risk_profile
        self.params = RISK_PARAMS[risk_profile]

    def analyze_holdings(
        self,
        holdings: list[dict],
        total_portfolio: float,
        cash_available: float,
        indicators_by_symbol: dict[str, dict[str, pd.DataFrame]],
        sentiment: SentimentReport | None = None,
    ) -> OpportunitiesReport:
        """Generate opportunities for all holdings.

        Args:
            holdings: list of dicts with keys:
                symbol, value, quantity, avg_cost, current_price,
                unrealized_pnl_pct, asset_type, source
            total_portfolio: total portfolio value in USD
            cash_available: cash/stablecoin available
            indicators_by_symbol: {symbol: {timeframe: DataFrame}}
            sentiment: current market sentiment
        """
        report = OpportunitiesReport(
            total_portfolio_value=total_portfolio,
            cash_available=cash_available,
        )
        report.market_regime = self._detect_regime(sentiment)

        for holding in holdings:
            symbol = holding["symbol"]
            value = holding.get("value", 0)
            alloc_pct = (value / total_portfolio * 100) if total_portfolio > 0 else 0

            if symbol.upper() in STABLECOINS or symbol == "USD":
                # Cash/stablecoins - look for put-selling or buy-the-dip opportunities
                self._opportunities_for_cash(report, holding, total_portfolio, sentiment)
                continue

            tf_data = indicators_by_symbol.get(symbol, {})
            if not tf_data:
                continue

            self._opportunities_for_position(
                report, holding, alloc_pct, total_portfolio, tf_data, sentiment
            )

        # Portfolio-level opportunities
        self._portfolio_level_opportunities(
            report, holdings, total_portfolio, cash_available, sentiment
        )

        return report

    def _detect_regime(self, sentiment: SentimentReport | None) -> str:
        if not sentiment or not sentiment.fear_greed:
            return "unknown"
        fg = sentiment.fear_greed.value
        if fg <= 20:
            return "capitulation"
        if fg <= 35:
            return "fear"
        if fg <= 55:
            return "neutral"
        if fg <= 75:
            return "greed"
        return "euphoria"

    def _opportunities_for_position(
        self,
        report: OpportunitiesReport,
        holding: dict,
        alloc_pct: float,
        total_portfolio: float,
        tf_data: dict[str, pd.DataFrame],
        sentiment: SentimentReport | None,
    ) -> None:
        symbol = holding["symbol"]
        value = holding.get("value", 0)
        pnl_pct = holding.get("unrealized_pnl_pct", 0)
        asset_type = holding.get("asset_type", "crypto")
        price = holding.get("current_price", 0)

        # Get key indicator readings from best available timeframe
        signals = self._extract_signals(tf_data)

        # --- 1. TAKE PROFIT (overbought + profitable) ---
        if signals["rsi"] and signals["rsi"] > 70 and pnl_pct > 10:
            sell_pct = 25 if pnl_pct > 50 else 15
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.TAKE_PROFIT,
                symbol=symbol,
                headline=f"Take {sell_pct}% profit on {symbol} - overbought (RSI {signals['rsi']:.0f})",
                details=(
                    f"{symbol} is up {pnl_pct:.1f}% and showing overbought conditions. "
                    f"Chart indicators suggest momentum may be fading. "
                    f"Lock in some gains while keeping upside exposure."
                ),
                action_steps=[
                    f"Sell {sell_pct}% of your {symbol} position (~${value * sell_pct / 100:,.0f})",
                    f"Convert to USDC or your preferred stablecoin",
                    f"Set a trailing stop on the remainder at ${price * 0.92:,.2f} (8% below)",
                    f"Consider re-entering on RSI pullback below 50",
                ],
                potential_gain=f"Lock in ${value * pnl_pct / 100 * sell_pct / 100:,.0f} profit",
                risk=f"Price continues higher and you miss {sell_pct}% of further gains",
                confidence=min(0.5 + signals["rsi"] / 200 + pnl_pct / 200, 0.9),
                priority=1 if signals["rsi"] > 80 else 2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 2. CUT LOSS (bearish trend + losing position) ---
        if pnl_pct < -15 and signals["trend_bearish"]:
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.CUT_LOSS,
                symbol=symbol,
                headline=f"Cut loss on {symbol} - down {pnl_pct:.1f}% with bearish trend",
                details=(
                    f"{symbol} is in a confirmed downtrend (price below 50 and 200 SMA, "
                    f"MACD bearish). Cutting now prevents further drawdown."
                ),
                action_steps=[
                    f"Sell 50-100% of your {symbol} position",
                    f"Move proceeds to USDC",
                    f"Set alert at prior support level to re-enter if trend reverses",
                ],
                potential_gain=f"Avoid further losses (currently down ${abs(value * pnl_pct / 100):,.0f})",
                risk="Price reverses immediately after you sell (whipsaw)",
                confidence=0.65 if pnl_pct < -25 else 0.5,
                priority=1,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 3. RIDE THE TREND (bullish, momentum strong) ---
        if signals["trend_bullish"] and signals["rsi"] and 40 < signals["rsi"] < 65:
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.RIDE_TREND,
                symbol=symbol,
                headline=f"Hold {symbol} - bullish trend, momentum healthy",
                details=(
                    f"{symbol} is in a confirmed uptrend with room to run. "
                    f"RSI at {signals['rsi']:.0f} shows momentum without being overbought. "
                    f"Keep your position and ride the trend."
                ),
                action_steps=[
                    f"Hold your {symbol} position (${value:,.0f})",
                    f"Set trailing stop at ${price * 0.90:,.2f} (10% below current)" if price else "Set a 10% trailing stop",
                    f"Consider adding on any pullback to the 50 SMA",
                ],
                potential_gain="Trend continuation toward next resistance",
                risk="Sudden reversal or macro event",
                confidence=0.6,
                priority=3,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 4. BUY THE DIP (oversold + in profit or near cost basis) ---
        if signals["rsi"] and signals["rsi"] < 30 and not signals["trend_bearish"]:
            add_pct = self.params["position_size_pct"]
            add_amount = total_portfolio * add_pct / 100
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.BUY_THE_DIP,
                symbol=symbol,
                headline=f"Add to {symbol} - RSI oversold at {signals['rsi']:.0f}",
                details=(
                    f"{symbol} is showing oversold conditions without being in a "
                    f"confirmed downtrend. This is often a buying opportunity. "
                    f"The long-term structure remains intact."
                ),
                action_steps=[
                    f"Buy ~${add_amount:,.0f} more {symbol} (up to {add_pct}% of portfolio)",
                    f"Set stop loss at ${price * 0.92:,.2f}" if price else "Set an 8% stop loss",
                    f"Scale in: buy 50% now, 50% if it dips another 3-5%",
                ],
                potential_gain="Mean reversion bounce of 5-15% typical from oversold RSI",
                risk=f"Oversold can become more oversold in a crash",
                confidence=0.55,
                priority=2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 5. ROTATE TO USDC (extreme greed + overbought + large position) ---
        regime = report.market_regime
        if regime in ("greed", "euphoria") and signals["rsi"] and signals["rsi"] > 65 and alloc_pct > 15:
            rotate_pct = 30 if regime == "euphoria" else 20
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.ROTATE_TO_USDC,
                symbol=symbol,
                headline=f"Rotate {rotate_pct}% of {symbol} to USDC - market euphoria + overbought",
                details=(
                    f"Market sentiment is at {regime} levels and {symbol} is overbought. "
                    f"This combination historically precedes corrections. "
                    f"Taking {rotate_pct}% off the table protects gains and gives you "
                    f"dry powder to buy back cheaper."
                ),
                action_steps=[
                    f"Sell {rotate_pct}% of {symbol} (~${value * rotate_pct / 100:,.0f})",
                    f"Convert to USDC",
                    f"Set buy-back alert at -10% to -15% from current price",
                    f"If price crashes >20%, use the USDC to buy back more than you sold",
                ],
                potential_gain=f"Protect ${value * rotate_pct / 100:,.0f} from potential 15-30% correction",
                risk="Market continues higher without you for the rotated portion",
                confidence=0.65 if regime == "euphoria" else 0.5,
                priority=1 if regime == "euphoria" else 2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 6. HEDGE WITH FUTURES (large position + some bearish signals) ---
        if (alloc_pct > 20 and signals["macd_bearish"]
                and asset_type == "crypto" and value > 1000):
            hedge_pct = 30
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.HEDGE_WITH_FUTURES,
                symbol=symbol,
                headline=f"Hedge {symbol} with {hedge_pct}% futures short - MACD turning bearish",
                details=(
                    f"Your {symbol} position is {alloc_pct:.1f}% of your portfolio "
                    f"and MACD is showing bearish momentum. A partial futures hedge "
                    f"protects downside while keeping your spot position and any "
                    f"staking/holding benefits."
                ),
                action_steps=[
                    f"Open a 1x short perpetual position on {symbol} worth ~${value * hedge_pct / 100:,.0f}",
                    f"Use an exchange that supports futures (Binance, Bybit, OKX)",
                    f"Set stop loss on the hedge at +5% above entry",
                    f"Close the hedge when MACD crosses bullish again",
                ],
                potential_gain=f"Offset up to {hedge_pct}% of losses if {symbol} drops 10-20%",
                risk="Funding rate costs + losses on the short if price pumps",
                confidence=0.55,
                priority=2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 7. SELL COVERED CALLS (stock or crypto with options support) ---
        if pnl_pct > 5 and signals["rsi"] and signals["rsi"] > 55 and value > 500:
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.SELL_COVERED_CALLS,
                symbol=symbol,
                headline=f"Sell covered calls on {symbol} for income",
                details=(
                    f"You hold {symbol} at a profit and it's not deeply oversold. "
                    f"Selling calls 10-15% above current price generates income while "
                    f"you hold. If called away, you sell at a profit. If not, you keep "
                    f"the premium."
                ),
                action_steps=[
                    f"Sell call options 10-15% OTM expiring 30-45 days out",
                    f"Strike target: ~${price * 1.12:,.2f}" if price else "Strike: 12% above current",
                    f"Collect premium income while holding the position",
                    f"Roll or let expire; repeat monthly",
                ] + ([
                    f"On Robinhood: go to the {symbol} page > Trade > Trade Options",
                ] if holding.get("source") == "robinhood" else [
                    f"Use Deribit or exchange options for crypto",
                ]),
                potential_gain="Monthly premium income of 1-3% of position value",
                risk=f"If {symbol} rallies past strike, shares get called away at that price",
                confidence=0.5,
                priority=3,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 8. BUY PUTS (large position + euphoric market) ---
        if alloc_pct > 10 and regime in ("greed", "euphoria") and value > 2000:
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.BUY_PUTS,
                symbol=symbol,
                headline=f"Buy puts on {symbol} as crash insurance",
                details=(
                    f"Market sentiment is at {regime} levels. Your {symbol} position "
                    f"is ${value:,.0f} ({alloc_pct:.1f}% of portfolio). Buying puts "
                    f"provides insurance against a sharp selloff at a fraction of the "
                    f"cost of selling the position."
                ),
                action_steps=[
                    f"Buy puts 10-15% OTM on {symbol} expiring 45-60 days out",
                    f"Budget 1-2% of position value (~${value * 0.015:,.0f}) for premium",
                    f"This protects against a >15% crash",
                ] + ([
                    f"On Robinhood: {symbol} page > Trade > Trade Options > Buy Put",
                ] if holding.get("source") == "robinhood" else []),
                potential_gain="3-10x return on premium if crash occurs (>15% drop)",
                risk=f"Lose the premium (~${value * 0.015:,.0f}) if market keeps rising",
                confidence=0.5,
                priority=2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 9. REBALANCE (over-concentrated) ---
        if alloc_pct > self.params["max_single_position"]:
            excess = alloc_pct - self.params["max_single_position"]
            trim_value = total_portfolio * excess / 100
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.REBALANCE,
                symbol=symbol,
                headline=f"Trim {symbol} - over-concentrated at {alloc_pct:.0f}% (limit: {self.params['max_single_position']}%)",
                details=(
                    f"Your {symbol} position exceeds your risk profile limit. "
                    f"Trimming by ~{excess:.1f}% spreads risk and frees capital "
                    f"for other opportunities."
                ),
                action_steps=[
                    f"Sell ~${trim_value:,.0f} of {symbol} to bring allocation to {self.params['max_single_position']}%",
                    f"Spread proceeds across other assets or hold in USDC",
                ],
                potential_gain="Reduced concentration risk, better risk-adjusted returns",
                risk="Trimmed asset outperforms in the short term",
                confidence=0.7,
                priority=2,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

        # --- 10. DCA ACCUMULATE (good long-term asset in fear market) ---
        if regime in ("fear", "capitulation") and not signals["trend_bearish"]:
            dca_amount = total_portfolio * 0.02  # 2% of portfolio
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.DCA_ACCUMULATE,
                symbol=symbol,
                headline=f"DCA into {symbol} - fear market discount",
                details=(
                    f"Market sentiment is at '{regime}' levels. Historically, "
                    f"accumulating quality assets during fear has been profitable. "
                    f"Dollar-cost averaging reduces timing risk."
                ),
                action_steps=[
                    f"Add ~${dca_amount:,.0f} to {symbol} now (2% of portfolio)",
                    f"Set recurring buys weekly until sentiment normalizes",
                    f"Keep total {symbol} allocation under {self.params['max_single_position']}%",
                ],
                potential_gain="Buy at discount; fear markets often precede strong recoveries",
                risk="Market could drop further before recovering",
                confidence=0.5,
                priority=3,
                current_value=value,
                allocation_pct=alloc_pct,
                unrealized_pnl_pct=pnl_pct,
            ))

    def _opportunities_for_cash(
        self,
        report: OpportunitiesReport,
        holding: dict,
        total_portfolio: float,
        sentiment: SentimentReport | None,
    ) -> None:
        """Generate opportunities for cash/stablecoin positions."""
        value = holding.get("value", 0)
        if value < 100:
            return

        regime = report.market_regime

        # Sell cash-secured puts during fear
        if regime in ("fear", "capitulation"):
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.SELL_CASH_SECURED_PUTS,
                symbol="PORTFOLIO",
                headline=f"Sell cash-secured puts - use ${value:,.0f} to collect fear premiums",
                details=(
                    f"With ${value:,.0f} in cash and market sentiment at '{regime}', "
                    f"put premiums are elevated. You can sell puts on assets you want "
                    f"to own, collecting premium income. If assigned, you buy at a "
                    f"discount. If not assigned, you keep the premium."
                ),
                action_steps=[
                    f"Pick 1-3 assets you'd want to own at 15-20% below current price",
                    f"Sell puts at those strike prices, 30-45 days out",
                    f"Keep ${value:,.0f} in reserve as collateral",
                    f"Good targets: top-10 crypto or blue-chip stocks in your watchlist",
                ],
                potential_gain="Premium income of 2-5% monthly (elevated by fear)",
                risk="Assigned at strike price in a continued downturn",
                confidence=0.55,
                priority=2,
                current_value=value,
            ))

        # Buy the dip with cash
        if regime in ("fear", "capitulation") and value > 500:
            deploy_pct = 30 if regime == "capitulation" else 15
            deploy_amount = value * deploy_pct / 100
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.BUY_THE_DIP,
                symbol="PORTFOLIO",
                headline=f"Deploy {deploy_pct}% of cash (~${deploy_amount:,.0f}) - fear market buying",
                details=(
                    f"Market is in '{regime}' mode. Deploying a portion of your "
                    f"${value:,.0f} cash position into quality assets at current "
                    f"fear-discounted prices could be very profitable."
                ),
                action_steps=[
                    f"Deploy ~${deploy_amount:,.0f} across your top 3-5 conviction assets",
                    f"Split evenly or weight toward the most oversold",
                    f"Keep the remaining ${value - deploy_amount:,.0f} as reserve",
                    f"Set stop losses at -10% on new purchases",
                ],
                potential_gain="Historically, buying fear has returned 20-50% within 6 months",
                risk="Market drops further before recovering; keep reserves",
                confidence=0.5,
                priority=2,
                current_value=value,
            ))

    def _portfolio_level_opportunities(
        self,
        report: OpportunitiesReport,
        holdings: list[dict],
        total_portfolio: float,
        cash: float,
        sentiment: SentimentReport | None,
    ) -> None:
        """Generate portfolio-wide opportunities."""
        # Check if portfolio is too exposed (low cash during greed)
        cash_pct = (cash / total_portfolio * 100) if total_portfolio > 0 else 0
        regime = report.market_regime

        if regime in ("greed", "euphoria") and cash_pct < 15:
            target_cash = total_portfolio * 0.20
            deficit = target_cash - cash
            report.opportunities.append(Opportunity(
                opp_type=OpportunityType.ROTATE_TO_USDC,
                symbol="PORTFOLIO",
                headline=f"Raise cash to 20% - only {cash_pct:.0f}% in stables during {regime}",
                details=(
                    f"Your cash allocation is {cash_pct:.1f}% during a {regime} market. "
                    f"Raising to 20% gives you ${deficit:,.0f} in dry powder to deploy "
                    f"when the inevitable correction comes."
                ),
                action_steps=[
                    f"Trim your most overbought/profitable positions by ~${deficit:,.0f} total",
                    f"Prioritize trimming positions that are most overbought (RSI > 70)",
                    f"Move proceeds to USDC",
                    f"Deploy this cash aggressively when sentiment drops below 30",
                ],
                potential_gain=f"${deficit:,.0f} ready to buy the dip at 15-30% discounts",
                risk="Market keeps rallying and you have less exposure",
                confidence=0.6,
                priority=2,
                current_value=cash,
                allocation_pct=cash_pct,
            ))

    def _extract_signals(self, tf_data: dict[str, pd.DataFrame]) -> dict:
        """Extract key signal readings from multi-timeframe data."""
        signals = {
            "rsi": None,
            "macd_bearish": False,
            "trend_bullish": False,
            "trend_bearish": False,
            "bb_pct": None,
            "vol_ratio": None,
        }

        # Prefer 4h for swing signals, fallback to 1d, then 1h
        for tf in ("4h", "1d", "1h"):
            if tf not in tf_data or tf_data[tf].empty:
                continue
            df = tf_data[tf]
            last = df.iloc[-1]

            if signals["rsi"] is None and "rsi_14" in df.columns:
                val = last.get("rsi_14")
                if val and not pd.isna(val):
                    signals["rsi"] = float(val)

            if "macd_hist" in df.columns:
                val = last.get("macd_hist")
                if val and not pd.isna(val) and val < 0:
                    signals["macd_bearish"] = True

            if "sma_50" in df.columns and "sma_200" in df.columns:
                sma50 = last.get("sma_50")
                sma200 = last.get("sma_200")
                close = last["close"]
                if sma50 and sma200 and not pd.isna(sma50) and not pd.isna(sma200):
                    if close > sma50 > sma200:
                        signals["trend_bullish"] = True
                    elif close < sma50 < sma200:
                        signals["trend_bearish"] = True

            if signals["bb_pct"] is None and "bb_pct" in df.columns:
                val = last.get("bb_pct")
                if val is not None and not pd.isna(val):
                    signals["bb_pct"] = float(val)

            if signals["vol_ratio"] is None and "vol_ratio" in df.columns:
                val = last.get("vol_ratio")
                if val and not pd.isna(val):
                    signals["vol_ratio"] = float(val)

        return signals
