"""Configuration management using pydantic-settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Robinhood
    robinhood_username: str = ""
    robinhood_password: str = ""
    robinhood_totp_secret: str = ""  # base32 secret for 2FA (from authenticator app setup)

    # Exchange keys
    binance_api_key: str = ""
    binance_secret: str = ""
    coinbase_api_key: str = ""
    coinbase_secret: str = ""
    coinbase_passphrase: str = ""
    kraken_api_key: str = ""
    kraken_secret: str = ""
    bybit_api_key: str = ""
    bybit_secret: str = ""

    # Data providers
    coingecko_api_key: str = ""

    # Alerts
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # General
    scan_interval: int = 300
    risk_profile: RiskProfile = RiskProfile.MODERATE
    alert_threshold: float = 1000.0
    stable_coin: str = "USDC"
    paper_trading: bool = True

    # Derived paths
    data_dir: Path = Field(default=Path("data"))
    log_dir: Path = Field(default=Path("logs"))


def load_settings() -> Settings:
    """Load settings from .env and environment variables."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings
