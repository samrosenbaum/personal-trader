# Personal Trader

Crypto and stock portfolio analyzer that connects to your **Robinhood** and **Coinbase** accounts (plus other exchanges), reads charts across multiple timeframes, and tells you exactly what you can do with what you hold to make money -- whether markets are going up or crashing.

## Features

- **Robinhood integration** - Pulls your full portfolio: crypto, stocks, ETFs, options, cash balance, P&L
- **Coinbase + exchange support** - Binance, Coinbase, Kraken, Bybit via API
- **"What can I do with what I hold?"** - The `opportunities` command analyzes every position and gives you specific, step-by-step actions
- **Technical analysis** - 40+ indicators (RSI, MACD, Bollinger, Ichimoku, ADX, etc.) across multiple timeframes
- **Chart pattern recognition** - Double top/bottom, head & shoulders, triangles, flags, candlestick patterns
- **Market sentiment** - Fear & Greed Index, funding rates, market cap trends
- **Derivatives advisor** - When to hedge with futures, buy puts for crash protection, sell puts for income
- **Portfolio risk management** - Concentration analysis, volatility tracking, diversification scoring
- **Continuous monitoring** - Background scanner with Telegram and Discord alerts

## Quick Start

```bash
# Install
cd personal-trader
pip install -e .

# Configure your accounts
cp .env.example .env
# Edit .env -- at minimum set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD
# For Coinbase, add COINBASE_API_KEY and COINBASE_SECRET

# See everything you hold
trader portfolio

# The main event: what should I do with my holdings?
trader opportunities

# Deep-dive into a specific asset
trader analyze BTC
trader analyze ETH

# Market scan
trader scan

# Continuous monitoring
trader watch
```

## Setting Up Robinhood

1. Add your Robinhood credentials to `.env`:
   ```
   ROBINHOOD_USERNAME=your_email@example.com
   ROBINHOOD_PASSWORD=your_password
   ```

2. For automatic 2FA (recommended):
   - In Robinhood app: Settings > Security > Two-Factor Authentication
   - Choose "Authentication App"
   - Instead of scanning the QR code, tap "Can't scan?" to reveal the secret key
   - Copy that base32 string and add to `.env`:
   ```
   ROBINHOOD_TOTP_SECRET=JBSWY3DPEHPK3PXP
   ```
   - Then scan the QR code with your authenticator app as well (so you have a backup)

3. Without the TOTP secret, Robinhood may send SMS codes which require manual entry.

## Commands

| Command | Description |
|---------|-------------|
| `trader portfolio` | Full portfolio view: Robinhood crypto/stocks/options + exchange holdings + risk analysis |
| `trader opportunities` | **The big one**: analyzes every position and tells you what to do to make money |
| `trader scan` | Scan watchlist symbols, show trading signals |
| `trader analyze BTC` | Deep technical analysis for a specific asset |
| `trader signals` | Top trading signals across all watched assets |
| `trader derivatives BTC` | Futures and options strategy recommendations |
| `trader sentiment` | Fear & Greed index, market overview, funding rates |
| `trader watch` | Continuous monitoring loop with alerts |
| `trader config` | Show current configuration (keys masked) |

## Opportunity Types

The `opportunities` command looks at each position you hold and generates specific ideas:

| Opportunity | When It Triggers | What It Tells You |
|-------------|-----------------|-------------------|
| **TAKE PROFIT** | RSI overbought + you're in profit | Sell X% at current price, set trailing stop on rest |
| **CUT LOSS** | Bearish trend + you're down >15% | Sell to stop bleeding, re-enter on reversal |
| **RIDE THE TREND** | Bullish trend + healthy momentum | Hold, set trailing stop, add on pullbacks |
| **BUY THE DIP** | RSI oversold + no confirmed downtrend | Add to position at discount |
| **ROTATE TO USDC** | Greed/euphoria + overbought + large position | Take chips off table, buy back after correction |
| **HEDGE WITH FUTURES** | MACD bearish + large position | Short futures to protect spot without selling |
| **SELL COVERED CALLS** | Holding at profit + not oversold | Sell calls above current price for income |
| **BUY PUTS** | Euphoric market + large position | Insurance against crash for a small premium |
| **SELL CASH-SECURED PUTS** | Fear market + you have cash | Collect premium; if assigned, buy at discount |
| **REBALANCE** | Single position exceeds risk limit | Trim to bring back within allocation limits |
| **DCA ACCUMULATE** | Fear/capitulation market | Dollar-cost average into quality assets at a discount |

## How It Works

### Signal Scoring

Signals are generated from three weighted layers:

1. **Technical Analysis (50%)** - Trend (MA crossovers, MACD, ADX, PSAR), Momentum (RSI, Stoch, MFI, CCI), Volatility (BB, squeeze), Volume (OBV, CMF, volume ratio)
2. **Chart Patterns (25%)** - Classical patterns weighted by confidence
3. **Market Sentiment (25%)** - Fear & Greed (contrarian), funding rates, market cap trends

### Risk Profiles

| Profile | Max Single Position | Min Stablecoins | Max Leverage |
|---------|-------------------|-----------------|-------------|
| Conservative | 15% | 30% | 1x |
| Moderate | 25% | 15% | 2x |
| Aggressive | 40% | 5% | 5x |

Set with `RISK_PROFILE=moderate` in `.env`.

### Derivatives Strategy

The advisor recommends derivatives when:
- **Futures hedging**: Sentiment > 65 (greedy) or volatility > 80% or bearish MA cross
- **Buy puts**: Greed + low vol (cheap insurance) or RSI > 75
- **Sell puts**: Fear < 30 with bullish technicals (collect elevated premium)
- **Funding arbitrage**: |funding rate| > 0.05% (delta-neutral carry trade)

## Supported Platforms

| Platform | Crypto | Stocks | Options | Futures |
|----------|--------|--------|---------|---------|
| **Robinhood** | Yes | Yes | Yes (read) | No |
| Coinbase | Yes | No | No | No |
| Binance | Yes | No | Yes | Yes |
| Kraken | Yes | No | No | Yes |
| Bybit | Yes | No | No | Yes |

## Architecture

```
personal_trader/
  config.py              # Settings from .env (Robinhood + exchanges)
  cli.py                 # CLI commands + Rich dashboard
  exchanges/
    robinhood.py         # Robinhood connector (robin_stocks)
    manager.py           # Multi-source portfolio aggregation
  analysis/
    technical.py         # 40+ indicators on OHLCV data
    patterns.py          # Chart pattern detection
    sentiment.py         # Fear/greed, funding, market overview
  strategy/
    opportunities.py     # "What can I do with what I hold?"
    signals.py           # Composite signal generator
    risk.py              # Portfolio risk + position sizing
    derivatives.py       # Futures/options strategy advisor
  monitoring/
    scanner.py           # Continuous market scanner
    alerts.py            # Telegram/Discord alert dispatch
```

## Important Notes

- This tool is **read-only** by default. It does not place trades on your behalf.
- `PAPER_TRADING=true` is the default. No real trades will be executed.
- Robinhood credentials are stored locally in your `.env` file. Never commit this file.
- All recommendations are informational. This is not financial advice. Do your own research.
