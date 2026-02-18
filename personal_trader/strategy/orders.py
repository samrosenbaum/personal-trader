"""Order planning helpers.

Builds actionable order plans (entry, stop, target) using signal levels and
portfolio-aware position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from personal_trader.config import RiskProfile
from personal_trader.strategy.risk import RiskManager

if TYPE_CHECKING:
    from personal_trader.strategy.signals import Signal


@dataclass
class OrderLeg:
    """A single order instruction."""

    side: str  # buy/sell
    order_type: str  # limit/stop/market
    price: float | None
    quantity: float | None
    note: str | None = None


@dataclass
class LadderStep:
    """A scale-in or scale-out ladder step."""

    price: float
    allocation_pct: float
    note: str


@dataclass
class OrderPlan:
    """A full order plan including sizing and bracket legs."""

    symbol: str
    action: str
    summary: str
    legs: list[OrderLeg] = field(default_factory=list)
    ladder: list[LadderStep] = field(default_factory=list)
    position_sizing: dict | None = None
    risk_notes: list[str] = field(default_factory=list)


class OrderPlanner:
    """Builds bracket-style orders using signal levels and risk sizing."""

    def __init__(self, risk_profile: RiskProfile) -> None:
        self.risk_manager = RiskManager(risk_profile)

    def build_plan(
        self,
        signal: "Signal",
        portfolio_value: float | None,
        atr: float | None = None,
    ) -> OrderPlan | None:
        """Create an order plan for a signal.

        Only build entry plans for BUY signals. SELL signals typically represent
        exiting/hedging, which depends on current holdings.
        """
        action_value = getattr(signal.action, "value", str(signal.action))
        if action_value not in ("buy", "strong_buy"):
            return None

        if signal.entry_price is None:
            return None

        sizing = None
        risk_notes = []
        if portfolio_value and signal.stop_loss is not None:
            sizing = self.risk_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
            )
            if "error" in sizing:
                risk_notes.append(sizing["error"])
                sizing = None
        elif portfolio_value:
            risk_notes.append("Stop loss unavailable; sizing skipped.")

        legs = [
            OrderLeg(
                side="buy",
                order_type="limit",
                price=signal.entry_price,
                quantity=sizing.get("units") if sizing else None,
                note="Entry limit",
            )
        ]

        if signal.stop_loss is not None:
            legs.append(OrderLeg(
                side="sell",
                order_type="stop",
                price=signal.stop_loss,
                quantity=sizing.get("units") if sizing else None,
                note="Protective stop",
            ))

        if signal.take_profit is not None:
            legs.append(OrderLeg(
                side="sell",
                order_type="limit",
                price=signal.take_profit,
                quantity=sizing.get("units") if sizing else None,
                note="Take profit",
            ))

        ladder = []
        if atr:
            ladder = [
                LadderStep(price=signal.entry_price, allocation_pct=25, note="Initial entry"),
                LadderStep(
                    price=round(signal.entry_price - atr * 0.5, 2),
                    allocation_pct=25,
                    note="Add on pullback (0.5 ATR)",
                ),
                LadderStep(
                    price=round(signal.entry_price + atr * 0.5, 2),
                    allocation_pct=50,
                    note="Add on breakout (0.5 ATR)",
                ),
            ]

        summary = "Bracket order: entry limit with stop-loss and take-profit."
        if not signal.stop_loss or not signal.take_profit:
            summary = "Entry limit with partial risk controls."

        return OrderPlan(
            symbol=signal.symbol,
            action=action_value,
            summary=summary,
            legs=legs,
            ladder=ladder,
            position_sizing=sizing,
            risk_notes=risk_notes,
        )
