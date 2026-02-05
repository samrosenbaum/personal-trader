# Personal Trader

Crypto portfolio analyzer and trading recommendation engine. Connects to your exchange accounts, reads charts across multiple timeframes, detects patterns, analyzes sentiment, and tells you when to buy, sell, move to USDC, hedge with futures, or trade options.

## Features

- **Multi-exchange portfolio tracking** - Binance, Coinbase, Kraken, Bybit
- **Technical analysis** - 40+ indicators (RSI, MACD, Bollinger, Ichimoku, ADX, etc.) across multiple timeframes (15m, 1h, 4h, 1d)
- **Chart pattern recognition** - Double top/bottom, head & shoulders, triangles, flags, candlestick patterns (doji, hammer, engulfing, morning/evening star)
- **Support & resistance detection** - Automated pivot-based level clustering
- **Market sentiment** - Fear & Greed Index, funding rates, market cap trends, trending coins
- **Trading signals** - Composite scoring from technicals + patterns + sentiment with specific buy/sell/hold recommendations
- **Derivatives advisor** - Futures hedging, put buying (crash protection), put selling (income), funding rate arbitrage
- **Portfolio risk management** - Concentration analysis, volatility tracking, diversification scoring, position sizing
- **Continuous monitoring** - Background scanner with configurable interval and alerts
- **Alerts** - Console, Telegram, and Discord notifications for urgent signals

## Quick Start

```bash
# Clone and install
cd personal-trader
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your exchange API keys

# Run commands
trader scan              # One-time market scan
trader analyze BTC       # Deep analysis of Bitcoin
trader portfolio         # View your portfolio + risk
trader signals           # All trading signals
trader derivatives BTC   # Futures/options strategies
trader sentiment         # Market sentiment dashboard
trader watch             # Continuous monitoring
trader config            # View configuration
```

## Commands

| Command | Description |
|---------|-------------|
| `trader scan` | Scan all watched symbols, show trading signals |
| `trader scan -s SOL/USDT` | Scan with additional symbols |
| `trader watch` | Continuous monitoring loop with alerts |
| `trader watch -i 60` | Watch with 60-second scan interval |
| `trader portfolio` | Portfolio breakdown + risk analysis |
| `trader analyze BTC` | Deep technical analysis for BTC |
| `trader analyze ETH -q USDC` | Analyze ETH/USDC pair |
| `trader signals` | Show top trading signals across watchlist |
| `trader derivatives BTC` | Derivatives strategy recommendations |
| `trader sentiment` | Fear & greed, market overview, funding rates |
| `trader config` | Show current configuration |

## Signal Types

The system generates these recommendation types:

| Signal | Meaning |
|--------|---------|
| **STRONG BUY** | Multiple bullish signals aligned across timeframes |
| **BUY** | Bullish bias, good entry conditions |
| **HOLD** | No clear direction, maintain current position |
| **SELL** | Bearish signals, reduce exposure |
| **STRONG SELL** | Multiple bearish signals, exit recommended |
| **MOVE TO STABLECOIN** | Risk-off: convert to USDC to protect capital |
| **HEDGE WITH FUTURES** | Open short futures to protect long spot positions |
| **BUY PUTS** | Purchase put options for crash protection |
| **SELL PUTS** | Sell puts for premium income (fearful markets) |

## How It Works

### Signal Generation

Signals are generated from three weighted layers:

1. **Technical Analysis (50%)** - Trend (MA crossovers, MACD, ADX, PSAR), Momentum (RSI, Stoch, MFI, CCI), Volatility (BB, squeeze), Volume (OBV, CMF, volume ratio)
2. **Chart Patterns (25%)** - Classical patterns weighted by confidence
3. **Market Sentiment (25%)** - Fear & Greed (contrarian), funding rates, market cap trends

Each layer produces a score from -100 to +100. The weighted composite determines the action.

### Risk Management

Portfolio risk is assessed on:
- Single position concentration vs risk profile limits
- Stablecoin allocation (cash buffer)
- Portfolio-level volatility (correlation-adjusted)
- Diversification (Herfindahl index)
- Per-asset drawdown tracking

### Derivatives Strategy

The advisor recommends derivatives when:
- **Futures hedging**: Sentiment > 65 (greedy) or volatility > 80% or bearish MA cross
- **Buy puts**: Greed + low vol (cheap insurance) or RSI > 75
- **Sell puts**: Fear < 30 with bullish technicals (collect elevated premium)
- **Funding arbitrage**: |funding rate| > 0.05% (delta-neutral carry trade)

## Configuration

Edit `.env` to configure:

- **Exchange API keys** - Read-only keys recommended for scanning
- **Risk profile** - `conservative`, `moderate`, or `aggressive`
- **Scan interval** - Seconds between continuous scans
- **Alert channels** - Telegram bot token/chat ID, Discord webhook
- **Paper trading** - Set `true` to disable any trade execution

## Supported Exchanges

| Exchange | Spot | Futures | Options |
|----------|------|---------|---------|
| Binance | Yes | Yes | Yes |
| Coinbase | Yes | No | No |
| Kraken | Yes | Yes | No |
| Bybit | Yes | Yes | No |

## Architecture

```
personal_trader/
  config.py              # Settings from .env
  cli.py                 # Click CLI commands + Rich dashboard
  exchanges/
    manager.py           # Multi-exchange connection via ccxt
  analysis/
    technical.py         # 40+ indicators on OHLCV data
    patterns.py          # Chart pattern detection
    sentiment.py         # Fear/greed, funding, market overview
  strategy/
    signals.py           # Composite signal generator
    risk.py              # Portfolio risk + position sizing
    derivatives.py       # Futures/options strategy advisor
  monitoring/
    scanner.py           # Continuous market scanner
    alerts.py            # Telegram/Discord alert dispatch
```
