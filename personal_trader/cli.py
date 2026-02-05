"""CLI interface for the personal trader.

Commands:
    trader scan          - Run a single market scan
    trader watch         - Start continuous monitoring
    trader portfolio     - View portfolio breakdown and risk
    trader analyze BTC   - Deep analysis of a specific asset
    trader signals       - Show latest trading signals
    trader derivatives   - Show derivatives strategy recommendations
    trader config        - Show current configuration
"""

from __future__ import annotations

import logging
import sys
import threading

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from personal_trader.config import load_settings

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_components(settings=None):
    """Initialize all components. Returns (settings, exchange_manager, scanner)."""
    from personal_trader.exchanges.manager import ExchangeManager
    from personal_trader.monitoring.alerts import AlertDispatcher
    from personal_trader.monitoring.scanner import MarketScanner

    if settings is None:
        settings = load_settings()
    em = ExchangeManager(settings)
    alert_disp = AlertDispatcher(settings)
    scanner = MarketScanner(settings, em, alert_disp)
    return settings, em, scanner


def _color_for_action(action_val: str) -> str:
    colors = {
        "strong_buy": "bold green",
        "buy": "green",
        "hold": "yellow",
        "sell": "red",
        "strong_sell": "bold red",
        "move_to_stablecoin": "bold magenta",
        "hedge_with_futures": "cyan",
        "sell_puts": "blue",
        "buy_puts": "magenta",
    }
    return colors.get(action_val, "white")


def _urgency_color(urgency_val: str) -> str:
    return {"immediate": "bold red", "high": "red", "medium": "yellow", "low": "green"}.get(
        urgency_val, "white"
    )


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """Personal Trader - Crypto portfolio analyzer and trading advisor."""
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@main.command()
@click.option("--symbols", "-s", multiple=True, help="Additional symbols to scan (e.g. BTC/USDT)")
def scan(symbols: tuple[str]) -> None:
    """Run a single market scan and display signals."""
    settings, em, scanner = _get_components()

    for s in symbols:
        scanner.add_to_watchlist(s)

    console.print(Panel("Running market scan...", style="bold cyan"))
    cycle = scanner.scan_once()

    # Summary
    console.print(f"\nScanned {len(cycle.results)} symbols at {cycle.timestamp.strftime('%H:%M:%S UTC')}")
    if cycle.portfolio_value > 0:
        console.print(f"Portfolio value: ${cycle.portfolio_value:,.2f}")
    console.print(f"Actionable signals: {cycle.actionable_count}\n")

    # Signals table
    table = Table(title="Trading Signals", show_lines=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Action", justify="center")
    table.add_column("Urgency", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Key Reasons")

    for result in sorted(cycle.results, key=lambda r: r.price, reverse=True):
        for sig in result.signals:
            action_text = Text(sig.action.value.upper(), style=_color_for_action(sig.action.value))
            urgency_text = Text(sig.urgency.value.upper(), style=_urgency_color(sig.urgency.value))
            reasons_str = "; ".join(sig.reasons[:2]) if sig.reasons else ""
            table.add_row(
                sig.symbol,
                f"${result.price:,.2f}" if result.price else "N/A",
                action_text,
                urgency_text,
                f"{sig.confidence:.0%}",
                reasons_str[:80],
            )

    console.print(table)

    if cycle.errors:
        console.print(f"\n[yellow]Warnings ({len(cycle.errors)}):[/yellow]")
        for err in cycle.errors[:5]:
            console.print(f"  - {err}")


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@main.command()
@click.option("--interval", "-i", default=None, type=int, help="Scan interval in seconds")
def watch(interval: int | None) -> None:
    """Start continuous market monitoring with alerts."""
    settings, em, scanner = _get_components()
    interval = interval or settings.scan_interval

    console.print(Panel(
        f"Starting continuous monitor\n"
        f"Interval: {interval}s | Risk profile: {settings.risk_profile.value}\n"
        f"Watching: {', '.join(scanner.watchlist[:5])}...\n"
        f"Press Ctrl+C to stop",
        title="Market Scanner",
        style="bold green",
    ))

    try:
        scanner.run_continuous(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Scanner stopped.[/yellow]")


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


@main.command()
def portfolio() -> None:
    """View portfolio breakdown and risk analysis."""
    from personal_trader.strategy.risk import RiskManager

    settings, em, scanner = _get_components()

    console.print(Panel("Fetching portfolio...", style="bold cyan"))
    pf = em.get_portfolio()

    if pf.total_usd == 0:
        console.print("[yellow]No portfolio data. Check your API keys in .env[/yellow]")
        return

    # Allocation table
    table = Table(title=f"Portfolio (${pf.total_usd:,.2f})")
    table.add_column("Asset", style="bold")
    table.add_column("Value (USD)", justify="right")
    table.add_column("Allocation %", justify="right")

    for asset, pct in sorted(pf.allocation.items(), key=lambda x: x[1], reverse=True):
        val = pf.balances.get(asset, 0)
        table.add_row(asset, f"${val:,.2f}", f"{pct:.1f}%")

    console.print(table)

    # Risk analysis
    risk_mgr = RiskManager(settings.risk_profile)
    report = risk_mgr.analyze_portfolio(pf.balances, pf.total_usd)

    risk_table = Table(title="Risk Analysis")
    risk_table.add_column("Metric", style="bold")
    risk_table.add_column("Value", justify="right")

    risk_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
    risk_table.add_row(
        "Overall Risk",
        Text(report.risk_level.value.upper(), style=risk_color.get(report.risk_level.value, "white")),
    )
    risk_table.add_row("Diversification Score", f"{report.diversification_score:.2f}")
    risk_table.add_row("Stablecoin %", f"{report.stablecoin_pct:.1f}%")
    risk_table.add_row("Top Concentration", f"{report.top_concentration_pct:.1f}%")
    if report.portfolio_volatility:
        risk_table.add_row("Portfolio Volatility", f"{report.portfolio_volatility:.1f}%")
    console.print(risk_table)

    if report.warnings:
        console.print("\n[bold red]Warnings:[/bold red]")
        for w in report.warnings:
            console.print(f"  [red]! {w}[/red]")

    if report.suggestions:
        console.print("\n[bold cyan]Suggestions:[/bold cyan]")
        for s in report.suggestions:
            console.print(f"  > {s}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@main.command()
@click.argument("asset")
@click.option("--quote", "-q", default="USDT", help="Quote currency")
def analyze(asset: str, quote: str) -> None:
    """Deep technical analysis of a specific asset."""
    from personal_trader.analysis.patterns import PatternDetector
    from personal_trader.analysis.technical import add_all_indicators, compute_multi_timeframe

    settings, em, scanner = _get_components()
    symbol = f"{asset.upper()}/{quote.upper()}"

    console.print(Panel(f"Analyzing {symbol}...", style="bold cyan"))

    # Multi-timeframe analysis
    tf_data = compute_multi_timeframe(
        fetch_fn=em.fetch_ohlcv,
        symbol=symbol,
    )

    if not tf_data:
        console.print(f"[red]Could not fetch data for {symbol}[/red]")
        return

    # Current price and key indicators
    for tf in ("1h", "4h", "1d"):
        if tf not in tf_data:
            continue
        df = tf_data[tf]
        last = df.iloc[-1]

        table = Table(title=f"{symbol} - {tf} Timeframe")
        table.add_column("Indicator", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Signal", justify="center")

        price = last["close"]
        table.add_row("Price", f"${price:,.2f}", "")

        # RSI
        if "rsi_14" in df.columns and last.get("rsi_14"):
            rsi = last["rsi_14"]
            rsi_sig = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            rsi_col = "green" if rsi < 30 else "red" if rsi > 70 else "yellow"
            table.add_row("RSI (14)", f"{rsi:.1f}", Text(rsi_sig, style=rsi_col))

        # MACD
        if "macd_hist" in df.columns:
            macd_h = last["macd_hist"]
            macd_sig = "Bullish" if macd_h > 0 else "Bearish"
            macd_col = "green" if macd_h > 0 else "red"
            table.add_row("MACD Histogram", f"{macd_h:.4f}", Text(macd_sig, style=macd_col))

        # Bollinger Band position
        if "bb_pct" in df.columns and last.get("bb_pct") is not None:
            bb = last["bb_pct"]
            bb_sig = "Upper band" if bb > 0.8 else "Lower band" if bb < 0.2 else "Middle"
            table.add_row("BB Position", f"{bb:.2f}", bb_sig)

        # Moving averages
        for period in (50, 200):
            sma_col = f"sma_{period}"
            if sma_col in df.columns and last.get(sma_col):
                sma_val = last[sma_col]
                above = "Above" if price > sma_val else "Below"
                color = "green" if price > sma_val else "red"
                table.add_row(f"SMA {period}", f"${sma_val:,.2f}", Text(above, style=color))

        # Volume ratio
        if "vol_ratio" in df.columns and last.get("vol_ratio"):
            vr = last["vol_ratio"]
            vr_sig = "High" if vr > 1.5 else "Low" if vr < 0.5 else "Normal"
            table.add_row("Volume Ratio", f"{vr:.2f}x", vr_sig)

        # ADX
        if "adx" in df.columns and last.get("adx"):
            adx = last["adx"]
            adx_sig = "Strong trend" if adx > 25 else "Weak/No trend"
            table.add_row("ADX", f"{adx:.1f}", adx_sig)

        console.print(table)
        console.print()

    # Pattern detection on daily
    if "1d" in tf_data:
        detector = PatternDetector(tf_data["1d"])
        report = detector.detect_all()

        if report.patterns:
            ptable = Table(title="Detected Patterns")
            ptable.add_column("Pattern", style="bold")
            ptable.add_column("Bias", justify="center")
            ptable.add_column("Confidence", justify="right")
            ptable.add_column("Description")

            for p in report.patterns:
                bias_color = "green" if p.bias.value == "bullish" else "red" if p.bias.value == "bearish" else "yellow"
                ptable.add_row(
                    p.pattern.value,
                    Text(p.bias.value.upper(), style=bias_color),
                    f"{p.confidence:.0%}",
                    p.description,
                )
            console.print(ptable)

        if report.support_levels:
            console.print(f"\n[green]Support levels:[/green] {', '.join(f'${s:,.2f}' for s in report.support_levels[:5])}")
        if report.resistance_levels:
            console.print(f"[red]Resistance levels:[/red] {', '.join(f'${r:,.2f}' for r in report.resistance_levels[:5])}")

    # Generate signals
    sentiment = scanner.sentiment_analyzer.get_full_report()
    signals = scanner.signal_gen.generate(
        symbol=symbol,
        indicators=tf_data,
        patterns=report if "1d" in tf_data else None,
        sentiment=sentiment,
    )

    if signals:
        console.print()
        sig_table = Table(title="Recommendations")
        sig_table.add_column("Action", style="bold")
        sig_table.add_column("Urgency")
        sig_table.add_column("Confidence", justify="right")
        sig_table.add_column("Entry", justify="right")
        sig_table.add_column("Stop Loss", justify="right")
        sig_table.add_column("Take Profit", justify="right")

        for sig in signals:
            sig_table.add_row(
                Text(sig.action.value.upper(), style=_color_for_action(sig.action.value)),
                Text(sig.urgency.value.upper(), style=_urgency_color(sig.urgency.value)),
                f"{sig.confidence:.0%}",
                f"${sig.entry_price:,.2f}" if sig.entry_price else "-",
                f"${sig.stop_loss:,.2f}" if sig.stop_loss else "-",
                f"${sig.take_profit:,.2f}" if sig.take_profit else "-",
            )
        console.print(sig_table)

        console.print("\n[bold]Reasons:[/bold]")
        for sig in signals:
            console.print(f"\n  [{_color_for_action(sig.action.value)}]{sig.action.value.upper()}[/]:")
            for r in sig.reasons:
                console.print(f"    - {r}")


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


@main.command()
def signals() -> None:
    """Show latest trading signals for all watched assets."""
    settings, em, scanner = _get_components()

    console.print(Panel("Generating signals for watchlist...", style="bold cyan"))
    cycle = scanner.scan_once()

    # Filter to most interesting signals
    all_signals = []
    for result in cycle.results:
        for sig in result.signals:
            all_signals.append((result, sig))

    # Sort by confidence
    all_signals.sort(key=lambda x: x[1].confidence, reverse=True)

    table = Table(title="Top Trading Signals", show_lines=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Action", justify="center")
    table.add_column("Urgency", justify="center")
    table.add_column("Conf.", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("Target", justify="right")

    for result, sig in all_signals[:20]:
        table.add_row(
            sig.symbol,
            f"${result.price:,.2f}" if result.price else "-",
            Text(sig.action.value.upper(), style=_color_for_action(sig.action.value)),
            Text(sig.urgency.value.upper(), style=_urgency_color(sig.urgency.value)),
            f"{sig.confidence:.0%}",
            f"${sig.entry_price:,.2f}" if sig.entry_price else "-",
            f"${sig.stop_loss:,.2f}" if sig.stop_loss else "-",
            f"${sig.take_profit:,.2f}" if sig.take_profit else "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------


@main.command()
@click.argument("asset", default="BTC")
@click.option("--quote", "-q", default="USDT", help="Quote currency")
def derivatives(asset: str, quote: str) -> None:
    """Show derivatives strategy recommendations for an asset."""
    from personal_trader.strategy.derivatives import DerivativesAdvisor

    settings, em, scanner = _get_components()
    symbol = f"{asset.upper()}/{quote.upper()}"

    console.print(Panel(f"Derivatives analysis for {symbol}...", style="bold cyan"))

    # Get current data
    try:
        ticker = em.fetch_ticker(symbol)
        price = ticker["last"]
    except Exception:
        console.print(f"[red]Could not fetch price for {symbol}[/red]")
        return

    # Get sentiment
    sentiment = scanner.sentiment_analyzer.get_full_report()
    sentiment_score = sentiment.sentiment_score if sentiment else 50

    # Get indicators
    from personal_trader.analysis.technical import add_all_indicators
    try:
        df_1d = em.fetch_ohlcv(symbol, "1d", 200)
        df_1d = add_all_indicators(df_1d)
    except Exception:
        df_1d = None

    # Get funding rate
    funding = None
    if sentiment and sentiment.funding_rates:
        for f in sentiment.funding_rates:
            if asset.upper() in f.symbol:
                funding = f.funding_rate
                break

    # Get volatility
    vol = None
    if df_1d is not None and len(df_1d) > 30:
        import numpy as np
        returns = df_1d["close"].pct_change().dropna().tail(30)
        vol = float(returns.std() * np.sqrt(365) * 100)

    advisor = DerivativesAdvisor(settings.risk_profile)
    report = advisor.analyze(
        symbol=symbol,
        spot_price=price,
        portfolio_allocation_pct=20,  # assume 20% if we don't know
        indicators=df_1d,
        sentiment_score=sentiment_score,
        funding_rate=funding,
        volatility_30d=vol,
    )

    # Display
    console.print(f"\n[bold]Market Conditions:[/bold] {report.market_conditions}")
    console.print(f"[bold]Recommended Hedge Ratio:[/bold] {report.overall_hedge_ratio:.0%}")
    console.print()

    if not report.has_recommendations:
        console.print("[green]No derivative strategies recommended at this time.[/green]")
        return

    for strat in report.strategies:
        panel_color = "red" if "short" in strat.direction else "green" if "long" in strat.direction else "cyan"
        lines = [
            f"[bold]Direction:[/bold] {strat.direction}",
            f"[bold]Rationale:[/bold] {strat.rationale}",
            f"[bold]Sizing:[/bold] {strat.sizing}",
            f"[bold]Entry:[/bold] {strat.entry_trigger}",
            f"[bold]Exit:[/bold] {strat.exit_trigger}",
            f"[bold]Max Loss:[/bold] {strat.max_loss_pct:.1f}% of portfolio",
            f"[bold]Expected P&L:[/bold] {strat.expected_pnl}",
            f"[bold]Risk/Reward:[/bold] {strat.risk_reward}",
        ]
        if strat.notes:
            lines.append(f"\n[bold]Notes:[/bold]")
            for note in strat.notes:
                lines.append(f"  - {note}")

        console.print(Panel(
            "\n".join(lines),
            title=f"{strat.strategy_type.value.upper()} - {strat.symbol}",
            style=panel_color,
        ))
        console.print()


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------


@main.command()
def sentiment() -> None:
    """Show current market sentiment analysis."""
    settings, em, scanner = _get_components()

    console.print(Panel("Fetching sentiment data...", style="bold cyan"))
    report = scanner.sentiment_analyzer.get_full_report()

    # Fear & Greed
    if report.fear_greed:
        fg = report.fear_greed
        color = "green" if fg.value < 30 else "red" if fg.value > 70 else "yellow"
        console.print(Panel(
            f"[bold {color}]{fg.value}/100 - {fg.label}[/bold {color}]\n\n"
            f"History (last 7 days): {', '.join(str(v) for _, v in fg.history[:7])}",
            title="Fear & Greed Index",
        ))

    # Market Overview
    if report.market_overview:
        mo = report.market_overview
        table = Table(title="Market Overview")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        cap_color = "green" if mo.market_cap_change_24h > 0 else "red"
        table.add_row("Total Market Cap", f"${mo.total_market_cap:,.0f}")
        table.add_row("24h Volume", f"${mo.total_volume_24h:,.0f}")
        table.add_row("24h Change", Text(f"{mo.market_cap_change_24h:+.2f}%", style=cap_color))
        table.add_row("BTC Dominance", f"{mo.btc_dominance:.1f}%")
        table.add_row("ETH Dominance", f"{mo.eth_dominance:.1f}%")
        table.add_row("Active Cryptos", f"{mo.active_cryptos:,}")
        console.print(table)

    # Trending
    if report.trending_coins:
        console.print(f"\n[bold]Trending:[/bold] {', '.join(report.trending_coins)}")

    # Funding rates
    if report.funding_rates:
        ftable = Table(title="Funding Rates")
        ftable.add_column("Symbol", style="bold")
        ftable.add_column("Rate", justify="right")
        ftable.add_column("Annualized", justify="right")

        for f in report.funding_rates:
            annual = f.funding_rate * 3 * 365 * 100
            rate_color = "green" if f.funding_rate < 0 else "red" if f.funding_rate > 0.0005 else "white"
            ftable.add_row(
                f.symbol,
                Text(f"{f.funding_rate*100:.4f}%", style=rate_color),
                f"{annual:.1f}%",
            )
        console.print(ftable)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@main.command("config")
def show_config() -> None:
    """Show current configuration (API keys masked)."""
    settings = load_settings()

    table = Table(title="Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    def mask(val: str) -> str:
        if not val:
            return "[dim]not set[/dim]"
        return val[:4] + "****" + val[-4:] if len(val) > 8 else "****"

    table.add_row("Risk Profile", settings.risk_profile.value)
    table.add_row("Scan Interval", f"{settings.scan_interval}s")
    table.add_row("Stable Coin", settings.stable_coin)
    table.add_row("Paper Trading", str(settings.paper_trading))
    table.add_row("Alert Threshold", f"${settings.alert_threshold:,.0f}")
    table.add_row("", "")
    table.add_row("Binance API", mask(settings.binance_api_key))
    table.add_row("Coinbase API", mask(settings.coinbase_api_key))
    table.add_row("Kraken API", mask(settings.kraken_api_key))
    table.add_row("Bybit API", mask(settings.bybit_api_key))
    table.add_row("", "")
    table.add_row("Telegram", mask(settings.telegram_bot_token))
    table.add_row("Discord", mask(settings.discord_webhook_url))

    console.print(table)


if __name__ == "__main__":
    main()
