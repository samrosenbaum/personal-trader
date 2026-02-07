"""Alert dispatching via multiple channels (console, Telegram, Discord)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import requests

from personal_trader.config import Settings
from personal_trader.strategy.signals import Signal

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Sends trading signal alerts to configured channels."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self._sent_cache: set[str] = set()  # Prevent duplicate alerts within session

    def send_signal_alert(self, signal: Signal) -> None:
        """Send an alert for a trading signal."""
        cache_key = f"{signal.symbol}:{signal.action.value}:{signal.timeframe}"
        if cache_key in self._sent_cache:
            return
        self._sent_cache.add(cache_key)

        message = self._format_signal(signal)

        # Always log to console
        logger.info(f"ALERT: {message}")

        # Telegram
        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            self._send_telegram(message)

        # Discord
        if self.settings.discord_webhook_url:
            self._send_discord(message)

    def _format_signal(self, signal: Signal) -> str:
        lines = [
            f"{'='*40}",
            f"SIGNAL: {signal.action.value.upper()}",
            f"Symbol: {signal.symbol}",
            f"Urgency: {signal.urgency.value.upper()}",
            f"Confidence: {signal.confidence:.0%}",
        ]
        if signal.entry_price:
            lines.append(f"Entry: ${signal.entry_price:,.2f}")
        if signal.stop_loss:
            lines.append(f"Stop Loss: ${signal.stop_loss:,.2f}")
        if signal.take_profit:
            lines.append(f"Take Profit: ${signal.take_profit:,.2f}")
        if signal.order_plan and signal.order_plan.position_sizing:
            sizing = signal.order_plan.position_sizing
            pct = sizing.get("position_pct")
            units = sizing.get("units")
            if pct is not None and units is not None:
                lines.append(f"Size: {pct:.1f}% ({units:.6f} units)")
        if signal.order_plan and signal.order_plan.ladder:
            ladder = ", ".join(
                f"{step.allocation_pct:.0f}% @ {step.price:.2f}"
                for step in signal.order_plan.ladder
            )
            lines.append(f"Ladder: {ladder}")
        lines.append(f"Reasons:")
        for r in signal.reasons:
            lines.append(f"  - {r}")
        lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"{'='*40}")
        return "\n".join(lines)

    def _send_telegram(self, message: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
            self.session.post(url, json={
                "chat_id": self.settings.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }, timeout=10)
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    def _send_discord(self, message: str) -> None:
        try:
            self.session.post(
                self.settings.discord_webhook_url,
                json={"content": f"```\n{message}\n```"},
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")

    def clear_cache(self) -> None:
        """Clear the sent-alerts cache (allow re-sending)."""
        self._sent_cache.clear()
