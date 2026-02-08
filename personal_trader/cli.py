"""CLI interface for the personal trader.

Commands:
    trader scan            - Run a single market scan
    trader watch           - Start continuous monitoring
    trader portfolio       - View portfolio breakdown and risk (Robinhood + exchanges)
    trader opportunities   - What can I do with what I hold to make money?
    trader analyze BTC     - Deep analysis of a specific asset
    trader signals         - Show latest trading signals
    trader history         - View trade history and realized P&L
    trader derivatives     - Show derivatives strategy recommendations
    trader config          - Show current configuration
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
    """View portfolio breakdown and risk analysis (Robinhood + exchanges)."""
    from personal_trader.strategy.risk import RiskManager

    settings, em, scanner = _get_components()

    console.print(Panel("Fetching portfolio from all sources...", style="bold cyan"))
    pf = em.get_portfolio()

    if pf.total_usd == 0:
        console.print("[yellow]No portfolio data. Check your API keys / Robinhood credentials in .env[/yellow]")
        return

    # --- Source breakdown ---
    if len(pf.exchange_breakdown) > 1 or "robinhood" in pf.exchange_breakdown:
        src_table = Table(title="Portfolio by Source")
        src_table.add_column("Source", style="bold")
        src_table.add_column("Value", justify="right")
        src_table.add_column("% of Total", justify="right")

        for source, balances in sorted(
            pf.exchange_breakdown.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        ):
            source_total = sum(balances.values())
            pct = (source_total / pf.total_usd * 100) if pf.total_usd > 0 else 0
            label = source.title()
            src_table.add_row(label, f"${source_total:,.2f}", f"{pct:.1f}%")

        src_table.add_row("", "", "")
        src_table.add_row("[bold]TOTAL[/bold]", f"[bold]${pf.total_usd:,.2f}[/bold]", "[bold]100%[/bold]")
        console.print(src_table)
        console.print()

    # --- Robinhood detail (crypto, stocks, options) ---
    rh_portfolio = em.get_robinhood_portfolio()
    if rh_portfolio and rh_portfolio.total_equity > 0:
        # Crypto
        if rh_portfolio.crypto_holdings:
            crypto_table = Table(title="Robinhood Crypto")
            crypto_table.add_column("Asset", style="bold")
            crypto_table.add_column("Qty", justify="right")
            crypto_table.add_column("Avg Cost", justify="right")
            crypto_table.add_column("Price", justify="right")
            crypto_table.add_column("Value", justify="right")
            crypto_table.add_column("P&L", justify="right")

            for h in sorted(rh_portfolio.crypto_holdings, key=lambda x: x.market_value, reverse=True):
                pnl_color = "green" if h.unrealized_pnl >= 0 else "red"
                crypto_table.add_row(
                    h.symbol,
                    f"{h.quantity:.6f}",
                    f"${h.avg_cost:,.2f}",
                    f"${h.current_price:,.2f}",
                    f"${h.market_value:,.2f}",
                    Text(f"{h.unrealized_pnl_pct:+.1f}%", style=pnl_color),
                )
            console.print(crypto_table)
            console.print()

        # Stocks
        if rh_portfolio.stock_holdings:
            stock_table = Table(title="Robinhood Stocks & ETFs")
            stock_table.add_column("Symbol", style="bold")
            stock_table.add_column("Qty", justify="right")
            stock_table.add_column("Avg Cost", justify="right")
            stock_table.add_column("Price", justify="right")
            stock_table.add_column("Value", justify="right")
            stock_table.add_column("P&L", justify="right")

            for h in sorted(rh_portfolio.stock_holdings, key=lambda x: x.market_value, reverse=True):
                pnl_color = "green" if h.unrealized_pnl >= 0 else "red"
                stock_table.add_row(
                    h.symbol,
                    f"{h.quantity:.2f}",
                    f"${h.avg_cost:,.2f}",
                    f"${h.current_price:,.2f}",
                    f"${h.market_value:,.2f}",
                    Text(f"{h.unrealized_pnl_pct:+.1f}%", style=pnl_color),
                )
            console.print(stock_table)
            console.print()

        # Options
        if rh_portfolio.option_positions:
            opt_table = Table(title="Robinhood Options")
            opt_table.add_column("Underlying", style="bold")
            opt_table.add_column("Type", justify="center")
            opt_table.add_column("Strike", justify="right")
            opt_table.add_column("Exp", justify="center")
            opt_table.add_column("Qty", justify="right")
            opt_table.add_column("Value", justify="right")
            opt_table.add_column("P&L", justify="right")

            for o in rh_portfolio.option_positions:
                pnl_color = "green" if o.unrealized_pnl >= 0 else "red"
                type_color = "green" if o.option_type == "call" else "red"
                opt_table.add_row(
                    o.symbol,
                    Text(f"{o.direction} {o.option_type}".upper(), style=type_color),
                    f"${o.strike:,.2f}",
                    o.expiration,
                    f"{o.quantity:.0f}",
                    f"${o.market_value:,.2f}",
                    Text(f"${o.unrealized_pnl:+,.2f}", style=pnl_color),
                )
            console.print(opt_table)
            console.print()

        # Cash
        if rh_portfolio.cash_balance > 0:
            console.print(f"Robinhood cash: ${rh_portfolio.cash_balance:,.2f}")
            console.print()

    # --- Combined allocation ---
    alloc_table = Table(title=f"Combined Allocation (${pf.total_usd:,.2f})")
    alloc_table.add_column("Asset", style="bold")
    alloc_table.add_column("Value (USD)", justify="right")
    alloc_table.add_column("Allocation %", justify="right")
    alloc_table.add_column("Source(s)")

    for asset, pct in sorted(pf.allocation.items(), key=lambda x: x[1], reverse=True):
        val = pf.balances.get(asset, 0)
        sources = [src for src, bals in pf.exchange_breakdown.items() if asset in bals]
        alloc_table.add_row(asset, f"${val:,.2f}", f"{pct:.1f}%", ", ".join(s.title() for s in sources))

    console.print(alloc_table)

    # --- Risk analysis ---
    risk_mgr = RiskManager(settings.risk_profile)
    cost_basis = {}
    try:
        cost_basis = em.get_cost_basis()
    except Exception:
        pass
    report = risk_mgr.analyze_portfolio(pf.balances, pf.total_usd, cost_basis=cost_basis)

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
    if report.var_95:
        risk_table.add_row("VaR 95% (1d)", f"${report.var_95:,.0f}")
    if report.cvar_95:
        risk_table.add_row("CVaR 95% (1d)", f"${report.cvar_95:,.0f}")
    console.print(risk_table)

    if report.stress_tests:
        console.print("\n[bold]Stress Tests:[/bold]")
        for scenario in report.stress_tests:
            console.print(f"  - {scenario}")

    if report.warnings:
        console.print("\n[bold red]Warnings:[/bold red]")
        for w in report.warnings:
            console.print(f"  [red]! {w}[/red]")

    if report.suggestions:
        console.print("\n[bold cyan]Suggestions:[/bold cyan]")
        for s in report.suggestions:
            console.print(f"  > {s}")


# ---------------------------------------------------------------------------
# opportunities
# ---------------------------------------------------------------------------


OPP_COLORS = {
    "ride_the_trend": "green",
    "take_profit": "bold yellow",
    "rotate_to_usdc": "bold magenta",
    "hedge_with_futures": "cyan",
    "buy_puts": "magenta",
    "sell_covered_calls": "blue",
    "sell_cash_secured_puts": "blue",
    "buy_the_dip": "bold green",
    "cut_loss": "bold red",
    "rebalance": "yellow",
    "dca_accumulate": "green",
    "funding_arbitrage": "cyan",
}


@main.command()
def opportunities() -> None:
    """What can I do with what I hold to make money?

    Analyzes every position in your portfolio (Robinhood + exchanges)
    and tells you exactly what actions you can take based on charts,
    sentiment, and your risk profile.
    """
    from personal_trader.analysis.technical import compute_multi_timeframe
    from personal_trader.strategy.opportunities import OpportunitiesEngine
    from personal_trader.strategy.risk import STABLECOINS

    settings, em, scanner = _get_components()

    console.print(Panel(
        "Scanning your holdings and charts to find money-making opportunities...",
        style="bold cyan",
    ))

    # 1. Gather all holdings from everywhere
    pf = em.get_portfolio()
    rh_portfolio = em.get_robinhood_portfolio()

    if pf.total_usd == 0:
        console.print("[yellow]No portfolio data. Configure Robinhood or exchange API keys in .env[/yellow]")
        return

    # Build unified holdings list with metadata
    holdings = []
    cash_total = 0.0

    # Robinhood holdings (with cost basis and P&L)
    if rh_portfolio:
        for h in rh_portfolio.crypto_holdings:
            holdings.append({
                "symbol": h.symbol,
                "value": h.market_value,
                "quantity": h.quantity,
                "avg_cost": h.avg_cost,
                "current_price": h.current_price,
                "unrealized_pnl_pct": h.unrealized_pnl_pct,
                "asset_type": "crypto",
                "source": "robinhood",
            })
        for h in rh_portfolio.stock_holdings:
            holdings.append({
                "symbol": h.symbol,
                "value": h.market_value,
                "quantity": h.quantity,
                "avg_cost": h.avg_cost,
                "current_price": h.current_price,
                "unrealized_pnl_pct": h.unrealized_pnl_pct,
                "asset_type": h.asset_type,
                "source": "robinhood",
            })
        cash_total += rh_portfolio.cash_balance

    # Fetch cost basis from trade history for exchange positions
    cost_basis = {}
    try:
        cost_basis = em.get_cost_basis()
    except Exception as e:
        console.print(f"  [dim]Could not fetch trade history for cost basis: {e}[/dim]")

    # Exchange holdings
    for source, balances in pf.exchange_breakdown.items():
        if source == "robinhood":
            continue  # already handled above
        for asset, value in balances.items():
            if asset.upper() in STABLECOINS or asset == "USD":
                cash_total += value
            else:
                # Merge with existing holding if from another source
                existing = next((h for h in holdings if h["symbol"] == asset), None)
                if existing:
                    existing["value"] += value
                else:
                    # Use cost basis from trade history if available
                    avg_cost = cost_basis.get(asset, 0)
                    # Estimate quantity and current price from value
                    current_price = 0.0
                    quantity = 0.0
                    try:
                        ticker = em.fetch_ticker(f"{asset}/USDT")
                        current_price = ticker["last"]
                        quantity = value / current_price if current_price > 0 else 0
                    except Exception:
                        pass
                    pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 and current_price > 0 else 0

                    holdings.append({
                        "symbol": asset,
                        "value": value,
                        "quantity": quantity,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "unrealized_pnl_pct": pnl_pct,
                        "asset_type": "crypto",
                        "source": source,
                    })

    # Add cash as a holding for cash-based opportunities
    if cash_total > 0:
        holdings.append({
            "symbol": "USD",
            "value": cash_total,
            "quantity": cash_total,
            "avg_cost": 1,
            "current_price": 1,
            "unrealized_pnl_pct": 0,
            "asset_type": "cash",
            "source": "mixed",
        })

    # 2. Fetch chart data for each non-cash holding
    console.print(f"Analyzing {len([h for h in holdings if h['asset_type'] != 'cash'])} positions...")
    indicators_by_symbol = {}
    for h in holdings:
        sym = h["symbol"]
        if sym == "USD" or sym.upper() in STABLECOINS:
            continue
        # For crypto, use USDT pair; for stocks, skip chart data (no ccxt data)
        if h["asset_type"] in ("stock", "etf"):
            continue  # stock chart data not available via ccxt
        pair = f"{sym}/USDT"
        try:
            tf_data = compute_multi_timeframe(
                fetch_fn=em.fetch_ohlcv,
                symbol=pair,
                timeframes=["1h", "4h", "1d"],
            )
            if tf_data:
                indicators_by_symbol[sym] = tf_data
        except Exception as e:
            console.print(f"  [dim]Could not fetch chart data for {sym}: {e}[/dim]")

    # 3. Get sentiment
    sentiment = scanner.sentiment_analyzer.get_full_report()

    # 4. Run the opportunities engine
    engine = OpportunitiesEngine(settings.risk_profile)
    report = engine.analyze_holdings(
        holdings=holdings,
        total_portfolio=pf.total_usd,
        cash_available=cash_total,
        indicators_by_symbol=indicators_by_symbol,
        sentiment=sentiment,
    )

    # 5. Display
    if not report.opportunities:
        console.print("[green]No specific opportunities found at this time. Portfolio looks stable.[/green]")
        return

    console.print(f"\nMarket regime: [bold]{report.market_regime}[/bold]")
    console.print(f"Found [bold]{len(report.opportunities)}[/bold] opportunities across your portfolio\n")

    for i, opp in enumerate(report.by_priority, 1):
        color = OPP_COLORS.get(opp.opp_type.value, "white")
        priority_label = {1: "DO NOW", 2: "SOON", 3: "CONSIDER"}.get(opp.priority, "")
        priority_color = {1: "bold red", 2: "yellow", 3: "dim"}.get(opp.priority, "white")

        # Build the panel content
        lines = []
        if opp.current_value > 0:
            lines.append(f"[bold]Position:[/bold] ${opp.current_value:,.0f} ({opp.allocation_pct:.1f}% of portfolio)")
        if opp.unrealized_pnl_pct:
            pnl_c = "green" if opp.unrealized_pnl_pct >= 0 else "red"
            lines.append(f"[bold]Unrealized P&L:[/bold] [{pnl_c}]{opp.unrealized_pnl_pct:+.1f}%[/{pnl_c}]")
        lines.append("")
        lines.append(opp.details)
        lines.append("")
        lines.append("[bold]Steps:[/bold]")
        for step in opp.action_steps:
            lines.append(f"  1. {step}" if opp.action_steps.index(step) == 0 else f"  {opp.action_steps.index(step)+1}. {step}")
        lines.append("")
        lines.append(f"[bold]Potential gain:[/bold] {opp.potential_gain}")
        lines.append(f"[bold]Risk:[/bold] {opp.risk}")
        lines.append(f"[bold]Confidence:[/bold] {opp.confidence:.0%}")

        title_str = (
            f"[{priority_color}][{priority_label}][/{priority_color}] "
            f"[{color}]{opp.opp_type.value.upper()}[/{color}] - {opp.symbol}"
        )

        console.print(Panel(
            "\n".join(lines),
            title=title_str,
            subtitle=opp.headline,
            style=color,
        ))
        console.print()


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
# history
# ---------------------------------------------------------------------------


@main.command()
@click.option("--symbol", "-s", default=None, help="Filter by symbol (e.g. BTC)")
@click.option("--days", "-d", default=90, type=int, help="Days of history to fetch")
@click.option("--source", default=None, help="Filter by source (robinhood, coinbase, etc.)")
@click.option("--limit", "-l", default=30, type=int, help="Number of recent trades to show")
def history(symbol: str | None, days: int, source: str | None, limit: int) -> None:
    """View trade history and realized P&L across all accounts."""
    from datetime import datetime, timedelta, timezone

    settings, em, _scanner = _get_components()

    since = datetime.now(timezone.utc) - timedelta(days=days)
    console.print(Panel(f"Fetching trade history (last {days} days)...", style="bold cyan"))

    trade_history = em.get_trade_history(since=since)

    if not trade_history.trades:
        console.print("[yellow]No trades found. Check your API credentials and connections.[/yellow]")
        return

    trades = trade_history.trades

    # Apply filters
    if symbol:
        symbol_upper = symbol.upper()
        trades = [t for t in trades if t.symbol == symbol_upper]
    if source:
        source_lower = source.lower()
        trades = [t for t in trades if t.source == source_lower]

    # --- Realized P&L Summary ---
    pnl_data = trade_history.realized_pnl
    if symbol:
        pnl_data = {k: v for k, v in pnl_data.items() if k == symbol.upper()}

    if pnl_data:
        pnl_table = Table(title="Realized P&L Summary", show_lines=True)
        pnl_table.add_column("Symbol", style="bold")
        pnl_table.add_column("Realized P&L", justify="right")
        pnl_table.add_column("P&L %", justify="right")
        pnl_table.add_column("Cost Basis", justify="right")
        pnl_table.add_column("Proceeds", justify="right")
        pnl_table.add_column("Avg Buy", justify="right")
        pnl_table.add_column("Avg Sell", justify="right")
        pnl_table.add_column("Trades", justify="right")

        for sym, pnl in sorted(pnl_data.items(), key=lambda x: abs(x[1].total_realized_pnl), reverse=True):
            pnl_color = "green" if pnl.total_realized_pnl >= 0 else "red"
            pnl_table.add_row(
                sym,
                Text(f"${pnl.total_realized_pnl:+,.2f}", style=pnl_color),
                Text(f"{pnl.pnl_pct:+.1f}%", style=pnl_color),
                f"${pnl.total_cost_basis:,.2f}",
                f"${pnl.total_proceeds:,.2f}",
                f"${pnl.avg_buy_price:,.2f}",
                f"${pnl.avg_sell_price:,.2f}",
                str(pnl.trade_count),
            )

        console.print(pnl_table)
        console.print()

    # --- Recent Trades ---
    recent = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:limit]

    if recent:
        trade_table = Table(title=f"Recent Trades (showing {len(recent)})", show_lines=True)
        trade_table.add_column("Date", style="dim")
        trade_table.add_column("Symbol", style="bold")
        trade_table.add_column("Side", justify="center")
        trade_table.add_column("Quantity", justify="right")
        trade_table.add_column("Price", justify="right")
        trade_table.add_column("Total", justify="right")
        trade_table.add_column("Fee", justify="right")
        trade_table.add_column("Source")

        for t in recent:
            side_color = "green" if t.side.value == "buy" else "red"
            trade_table.add_row(
                t.timestamp.strftime("%Y-%m-%d %H:%M"),
                t.symbol,
                Text(t.side.value.upper(), style=side_color),
                f"{t.quantity:.6f}",
                f"${t.price:,.2f}",
                f"${t.total:,.2f}",
                f"${t.fee:.2f}" if t.fee > 0 else "-",
                t.source.title(),
            )

        console.print(trade_table)

    # --- Summary ---
    total_pnl_color = "green" if trade_history.total_realized_pnl >= 0 else "red"
    summary_lines = [
        f"Total trades: {len(trades)}",
        f"Total realized P&L: [{total_pnl_color}]${trade_history.total_realized_pnl:+,.2f}[/{total_pnl_color}]",
        f"Total fees paid: ${trade_history.total_fees:,.2f}",
    ]

    # Show current cost basis
    cost_basis = em.get_cost_basis()
    if cost_basis:
        cb_filtered = cost_basis if not symbol else {k: v for k, v in cost_basis.items() if k == symbol.upper()}
        if cb_filtered:
            summary_lines.append("")
            summary_lines.append("[bold]Current Cost Basis (avg per unit):[/bold]")
            for sym, avg_cost in sorted(cb_filtered.items()):
                summary_lines.append(f"  {sym}: ${avg_cost:,.2f}")

    console.print(Panel("\n".join(summary_lines), title="Summary", border_style="cyan"))


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
    table.add_row("[bold]Robinhood[/bold]", mask(settings.robinhood_username))
    table.add_row("Robinhood 2FA", "configured" if settings.robinhood_totp_secret else "[dim]not set[/dim]")
    table.add_row("", "")
    table.add_row("Coinbase API", mask(settings.coinbase_api_key))
    table.add_row("Binance API", mask(settings.binance_api_key))
    table.add_row("Kraken API", mask(settings.kraken_api_key))
    table.add_row("Bybit API", mask(settings.bybit_api_key))
    table.add_row("", "")
    table.add_row("Telegram", mask(settings.telegram_bot_token))
    table.add_row("Discord", mask(settings.discord_webhook_url))

    console.print(table)


if __name__ == "__main__":
    main()
