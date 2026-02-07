"""Portfolio risk management and position sizing.

Tracks portfolio exposure, calculates risk metrics, and determines
safe position sizes for trades and hedges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from personal_trader.config import RiskProfile

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionRisk:
    symbol: str
    allocation_pct: float
    unrealized_pnl_pct: float
    volatility_30d: float | None
    max_drawdown_30d: float | None
    risk_level: RiskLevel
    recommendation: str


@dataclass
class PortfolioRiskReport:
    total_value_usd: float
    risk_level: RiskLevel
    diversification_score: float  # 0-1 (1 = well diversified)
    stablecoin_pct: float
    top_concentration_pct: float  # largest single position %
    portfolio_volatility: float | None
    var_95: float | None = None
    cvar_95: float | None = None
    positions: list[PositionRisk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    stress_tests: list[str] = field(default_factory=list)


STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FRAX", "LUSD", "USD"}

# Risk thresholds per profile
RISK_PARAMS = {
    RiskProfile.CONSERVATIVE: {
        "max_single_position": 15,   # max % in one asset
        "min_stablecoin": 30,        # min % in stables
        "max_leverage": 1.0,
        "stop_loss_pct": 5,
        "position_size_pct": 5,      # max % of portfolio per trade
    },
    RiskProfile.MODERATE: {
        "max_single_position": 25,
        "min_stablecoin": 15,
        "max_leverage": 2.0,
        "stop_loss_pct": 8,
        "position_size_pct": 10,
    },
    RiskProfile.AGGRESSIVE: {
        "max_single_position": 40,
        "min_stablecoin": 5,
        "max_leverage": 5.0,
        "stop_loss_pct": 12,
        "position_size_pct": 20,
    },
}


class RiskManager:
    """Analyses portfolio risk and provides sizing recommendations."""

    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE) -> None:
        self.risk_profile = risk_profile
        self.params = RISK_PARAMS[risk_profile]

    def analyze_portfolio(
        self,
        balances: dict[str, float],
        total_usd: float,
        price_histories: dict[str, pd.DataFrame] | None = None,
    ) -> PortfolioRiskReport:
        """Full portfolio risk analysis.

        Args:
            balances: asset -> USD value
            total_usd: total portfolio value
            price_histories: asset -> DataFrame with 'close' column (for volatility)
        """
        if total_usd == 0:
            return PortfolioRiskReport(
                total_value_usd=0,
                risk_level=RiskLevel.LOW,
                diversification_score=0,
                stablecoin_pct=0,
                top_concentration_pct=0,
                portfolio_volatility=None,
            )

        price_histories = price_histories or {}
        positions = []
        allocations = {}
        stablecoin_value = 0.0

        for asset, usd_val in balances.items():
            if usd_val < 0.01:
                continue
            alloc_pct = (usd_val / total_usd) * 100
            allocations[asset] = alloc_pct

            if asset.upper() in STABLECOINS:
                stablecoin_value += usd_val

            vol_30d = None
            max_dd = None
            if asset in price_histories and len(price_histories[asset]) > 5:
                hist = price_histories[asset]
                returns = hist["close"].pct_change().dropna()
                vol_30d = float(returns.tail(30).std() * np.sqrt(365) * 100) if len(returns) >= 30 else None
                max_dd = self._max_drawdown(hist["close"].values)

            risk_level = self._classify_position_risk(alloc_pct, vol_30d)
            rec = self._position_recommendation(asset, alloc_pct, vol_30d, risk_level)

            positions.append(PositionRisk(
                symbol=asset,
                allocation_pct=round(alloc_pct, 2),
                unrealized_pnl_pct=0,  # would need cost basis data
                volatility_30d=round(vol_30d, 2) if vol_30d else None,
                max_drawdown_30d=round(max_dd, 2) if max_dd else None,
                risk_level=risk_level,
                recommendation=rec,
            ))

        stablecoin_pct = (stablecoin_value / total_usd) * 100 if total_usd > 0 else 0
        top_concentration = max(allocations.values()) if allocations else 0
        diversification = self._diversification_score(allocations)
        portfolio_vol = self._portfolio_volatility(balances, total_usd, price_histories)
        var_95, cvar_95 = self._portfolio_var_cvar(balances, total_usd, price_histories)

        overall_risk = self._overall_risk(stablecoin_pct, top_concentration, portfolio_vol)

        report = PortfolioRiskReport(
            total_value_usd=round(total_usd, 2),
            risk_level=overall_risk,
            diversification_score=round(diversification, 2),
            stablecoin_pct=round(stablecoin_pct, 2),
            top_concentration_pct=round(top_concentration, 2),
            portfolio_volatility=round(portfolio_vol, 2) if portfolio_vol else None,
            var_95=round(var_95, 2) if var_95 is not None else None,
            cvar_95=round(cvar_95, 2) if cvar_95 is not None else None,
            positions=sorted(positions, key=lambda p: p.allocation_pct, reverse=True),
        )

        # Generate warnings
        if stablecoin_pct < self.params["min_stablecoin"]:
            report.warnings.append(
                f"Stablecoin allocation ({stablecoin_pct:.1f}%) below recommended "
                f"minimum ({self.params['min_stablecoin']}%) for {self.risk_profile.value} profile"
            )
        if top_concentration > self.params["max_single_position"]:
            top_asset = max(allocations, key=allocations.get)
            report.warnings.append(
                f"{top_asset} concentration ({top_concentration:.1f}%) exceeds "
                f"limit ({self.params['max_single_position']}%)"
            )
        if overall_risk == RiskLevel.CRITICAL:
            report.warnings.append("Portfolio risk is CRITICAL - consider immediate rebalancing")

        # Generate suggestions
        report.suggestions = self._generate_suggestions(report)
        report.stress_tests = self._stress_test_scenarios(balances, total_usd, price_histories)

        return report

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> dict:
        """Calculate recommended position size based on risk parameters.

        Returns dict with size info and risk/reward details.
        """
        risk_amount = portfolio_value * (self.params["position_size_pct"] / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return {"error": "Stop loss cannot equal entry price"}

        units = risk_amount / risk_per_unit
        position_value = units * entry_price
        position_pct = (position_value / portfolio_value) * 100

        return {
            "units": round(units, 6),
            "position_value_usd": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            "risk_amount_usd": round(risk_amount, 2),
            "risk_pct": self.params["position_size_pct"],
            "stop_loss": stop_loss_price,
            "entry": entry_price,
        }

    def adjusted_max_leverage(self, portfolio_volatility: float | None) -> float:
        """Dynamically adjust max leverage based on volatility regime."""
        base = self.params["max_leverage"]
        if portfolio_volatility is None:
            return base
        if portfolio_volatility > 80:
            return max(1.0, base * 0.5)
        if portfolio_volatility > 50:
            return base * 0.75
        return base

    @staticmethod
    def _max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _portfolio_var_cvar(
        self,
        balances: dict[str, float],
        total_usd: float,
        price_histories: dict[str, pd.DataFrame],
    ) -> tuple[float | None, float | None]:
        """Compute 95% VaR/CVaR using historical returns."""
        if total_usd == 0:
            return None, None
        returns = self._portfolio_returns(balances, total_usd, price_histories)
        if returns is None or returns.empty:
            return None, None
        var_level = np.percentile(returns, 5)
        cvar = returns[returns <= var_level].mean()
        return float(abs(var_level) * total_usd), float(abs(cvar) * total_usd)

    def _portfolio_returns(
        self,
        balances: dict[str, float],
        total_usd: float,
        price_histories: dict[str, pd.DataFrame],
    ) -> pd.Series | None:
        if not price_histories:
            return None
        weighted = []
        for asset, usd_val in balances.items():
            if asset not in price_histories or usd_val <= 0:
                continue
            hist = price_histories[asset]
            if len(hist) < 30 or "close" not in hist:
                continue
            returns = hist["close"].pct_change().dropna()
            weight = usd_val / total_usd
            weighted.append(returns * weight)
        if not weighted:
            return None
        return sum(weighted)

    def _stress_test_scenarios(
        self,
        balances: dict[str, float],
        total_usd: float,
        price_histories: dict[str, pd.DataFrame],
    ) -> list[str]:
        """Simple stress scenarios based on dominant assets."""
        scenarios = []
        if total_usd == 0:
            return scenarios
        top_assets = sorted(balances.items(), key=lambda x: x[1], reverse=True)[:3]
        for asset, usd_val in top_assets:
            loss = usd_val * 0.2
            scenarios.append(f"{asset} -20% shock: -${loss:,.0f}")
        if price_histories:
            returns = self._portfolio_returns(balances, total_usd, price_histories)
            if returns is not None and not returns.empty:
                worst = returns.min() * total_usd
                scenarios.append(f"Worst 1-day historical loss: -${abs(worst):,.0f}")
        return scenarios

    @staticmethod
    def _diversification_score(allocations: dict[str, float]) -> float:
        """Herfindahl-based diversification score. 1 = well diversified, 0 = concentrated."""
        if not allocations:
            return 0
        weights = np.array(list(allocations.values())) / 100
        hhi = np.sum(weights**2)
        n = len(allocations)
        if n <= 1:
            return 0
        min_hhi = 1 / n
        return round(1 - (hhi - min_hhi) / (1 - min_hhi), 4) if (1 - min_hhi) > 0 else 0

    def _classify_position_risk(self, alloc_pct: float, volatility: float | None) -> RiskLevel:
        if alloc_pct > self.params["max_single_position"] * 1.5:
            return RiskLevel.CRITICAL
        if alloc_pct > self.params["max_single_position"]:
            return RiskLevel.HIGH
        if volatility and volatility > 100:
            return RiskLevel.HIGH
        if volatility and volatility > 60:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _position_recommendation(
        self, asset: str, alloc_pct: float, vol: float | None, risk: RiskLevel
    ) -> str:
        if asset.upper() in STABLECOINS:
            return "Hold - stablecoin position"
        if risk == RiskLevel.CRITICAL:
            return f"Reduce immediately - overexposed at {alloc_pct:.1f}%"
        if risk == RiskLevel.HIGH:
            return f"Consider trimming - high risk at {alloc_pct:.1f}%"
        if vol and vol > 80:
            return "Monitor closely - high volatility asset"
        return "Hold - within risk parameters"

    def _portfolio_volatility(
        self,
        balances: dict[str, float],
        total_usd: float,
        histories: dict[str, pd.DataFrame],
    ) -> float | None:
        """Estimate portfolio-level annualized volatility."""
        if not histories or total_usd == 0:
            return None

        returns_data = {}
        weights = {}
        for asset, hist in histories.items():
            if asset in balances and len(hist) > 30:
                ret = hist["close"].pct_change().dropna().tail(30)
                if len(ret) > 5:
                    returns_data[asset] = ret
                    weights[asset] = balances.get(asset, 0) / total_usd

        if len(returns_data) < 2:
            # Single asset, just return its vol
            if returns_data:
                only = list(returns_data.values())[0]
                return float(only.std() * np.sqrt(365) * 100)
            return None

        # Weighted portfolio variance
        aligned = pd.DataFrame(returns_data).dropna()
        if len(aligned) < 5:
            return None

        w = np.array([weights.get(c, 0) for c in aligned.columns])
        w = w / w.sum()
        cov = aligned.cov().values
        port_var = w @ cov @ w
        return float(np.sqrt(port_var) * np.sqrt(365) * 100)

    def _overall_risk(
        self, stablecoin_pct: float, top_concentration: float, vol: float | None
    ) -> RiskLevel:
        score = 0
        if stablecoin_pct < 5:
            score += 3
        elif stablecoin_pct < 15:
            score += 2
        elif stablecoin_pct < 30:
            score += 1

        if top_concentration > 50:
            score += 3
        elif top_concentration > 35:
            score += 2
        elif top_concentration > 25:
            score += 1

        if vol and vol > 120:
            score += 3
        elif vol and vol > 80:
            score += 2
        elif vol and vol > 50:
            score += 1

        if score >= 7:
            return RiskLevel.CRITICAL
        if score >= 4:
            return RiskLevel.HIGH
        if score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_suggestions(self, report: PortfolioRiskReport) -> list[str]:
        suggestions = []
        if report.stablecoin_pct < self.params["min_stablecoin"]:
            deficit = self.params["min_stablecoin"] - report.stablecoin_pct
            suggestions.append(
                f"Move ~{deficit:.1f}% of portfolio to stablecoins (USDC recommended)"
            )

        over_allocated = [
            p for p in report.positions
            if p.allocation_pct > self.params["max_single_position"]
            and p.symbol.upper() not in STABLECOINS
        ]
        for pos in over_allocated:
            excess = pos.allocation_pct - self.params["max_single_position"]
            suggestions.append(
                f"Trim {pos.symbol} by ~{excess:.1f}% to reduce concentration risk"
            )

        high_vol = [
            p for p in report.positions
            if p.volatility_30d and p.volatility_30d > 100
            and p.symbol.upper() not in STABLECOINS
        ]
        for pos in high_vol:
            suggestions.append(
                f"Consider hedging {pos.symbol} (vol: {pos.volatility_30d}%) with futures or reducing size"
            )

        if report.portfolio_volatility is not None:
            adjusted = self.adjusted_max_leverage(report.portfolio_volatility)
            if adjusted < self.params["max_leverage"]:
                suggestions.append(
                    f"High volatility detected; reduce max leverage to ~{adjusted:.1f}x"
                )

        if not suggestions:
            suggestions.append("Portfolio is within risk parameters - no immediate action needed")

        return suggestions
