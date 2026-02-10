"""Alert dispatching via multiple channels (console, Telegram, Discord, Email)."""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

        # Email
        if self.settings.email_smtp_host and self.settings.email_recipient:
            self._send_email(signal, message)

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

    def _send_email(self, signal: Signal, plain_text: str) -> None:
        """Send alert via SMTP email with HTML formatting."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"[{signal.urgency.value.upper()}] "
                f"{signal.action.value.upper()} {signal.symbol}"
            )
            msg["From"] = self.settings.email_username
            msg["To"] = self.settings.email_recipient

            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(self._format_email_html(signal), "html"))

            with smtplib.SMTP(
                self.settings.email_smtp_host,
                self.settings.email_smtp_port,
                timeout=15,
            ) as server:
                server.starttls()
                server.login(
                    self.settings.email_username,
                    self.settings.email_password,
                )
                server.send_message(msg)

            logger.info(f"Email alert sent to {self.settings.email_recipient}")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _format_email_html(self, signal: Signal) -> str:
        """Build an HTML email body from a Signal."""
        action_colors = {
            "strong_buy": "#22c55e", "buy": "#4ade80",
            "hold": "#eab308",
            "sell": "#f87171", "strong_sell": "#ef4444",
            "move_to_stablecoin": "#c084fc",
            "hedge_with_futures": "#22d3ee",
            "sell_puts": "#60a5fa", "buy_puts": "#a78bfa",
        }
        urgency_colors = {
            "immediate": "#ef4444", "high": "#f87171",
            "medium": "#eab308", "low": "#4ade80",
        }
        action_color = action_colors.get(signal.action.value, "#ffffff")
        urgency_color = urgency_colors.get(signal.urgency.value, "#ffffff")

        reasons_html = "".join(f"<li>{r}</li>" for r in signal.reasons)

        price_rows = ""
        if signal.entry_price:
            price_rows += (
                f'<tr><td style="padding:4px 8px;font-weight:bold;">Entry</td>'
                f'<td style="padding:4px 8px;">${signal.entry_price:,.2f}</td></tr>'
            )
        if signal.stop_loss:
            price_rows += (
                f'<tr><td style="padding:4px 8px;font-weight:bold;">Stop Loss</td>'
                f'<td style="padding:4px 8px;">${signal.stop_loss:,.2f}</td></tr>'
            )
        if signal.take_profit:
            price_rows += (
                f'<tr><td style="padding:4px 8px;font-weight:bold;">Take Profit</td>'
                f'<td style="padding:4px 8px;">${signal.take_profit:,.2f}</td></tr>'
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return f"""<html><body style="font-family:Arial,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px;">
<div style="max-width:600px;margin:0 auto;background:#16213e;border-radius:8px;padding:20px;">
  <h2 style="margin:0 0 16px;color:#e0e0e0;">Trading Alert: {signal.symbol}</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
    <tr>
      <td style="padding:4px 8px;font-weight:bold;">Action</td>
      <td style="padding:4px 8px;"><span style="color:{action_color};font-weight:bold;">{signal.action.value.upper()}</span></td>
    </tr>
    <tr>
      <td style="padding:4px 8px;font-weight:bold;">Urgency</td>
      <td style="padding:4px 8px;"><span style="color:{urgency_color};font-weight:bold;">{signal.urgency.value.upper()}</span></td>
    </tr>
    <tr>
      <td style="padding:4px 8px;font-weight:bold;">Confidence</td>
      <td style="padding:4px 8px;">{signal.confidence:.0%}</td>
    </tr>
    {price_rows}
  </table>
  <h3 style="color:#e0e0e0;margin:12px 0 8px;">Reasons</h3>
  <ul style="margin:0;padding-left:20px;">{reasons_html}</ul>
  <p style="color:#888;font-size:12px;margin-top:16px;">{timestamp} | personal-trader</p>
</div>
</body></html>"""

    def clear_cache(self) -> None:
        """Clear the sent-alerts cache (allow re-sending)."""
        self._sent_cache.clear()
