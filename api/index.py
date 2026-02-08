"""Vercel serverless entry point — minimal API for the personal-trader CLI."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    """Health / status endpoint so Vercel deployments succeed."""

    def do_GET(self):
        has_robinhood = bool(os.environ.get("ROBINHOOD_USERNAME"))
        has_coinbase = bool(os.environ.get("COINBASE_API_KEY"))
        has_binance = bool(os.environ.get("BINANCE_API_KEY"))

        configured_exchanges = []
        if has_robinhood:
            configured_exchanges.append("robinhood")
        if has_coinbase:
            configured_exchanges.append("coinbase")
        if has_binance:
            configured_exchanges.append("binance")

        body = json.dumps({
            "name": "personal-trader",
            "version": "0.1.0",
            "status": "ok",
            "description": "Crypto & stock portfolio analyzer with trading recommendations",
            "cli_usage": "pip install -e . && trader --help",
            "configured_exchanges": configured_exchanges,
            "endpoints": {
                "/api/health": "Health check",
                "/api/portfolio": "Portfolio summary (coming soon)",
            },
        }, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
