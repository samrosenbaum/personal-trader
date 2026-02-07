"""Derivatives and volatility surface analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests


DERIBIT_URL = "https://www.deribit.com/api/v2"


@dataclass
class VolSurfacePoint:
    tenor_days: int
    atm_iv: float | None
    call_25_delta_iv: float | None
    put_25_delta_iv: float | None

    @property
    def skew(self) -> float | None:
        if self.call_25_delta_iv is None or self.put_25_delta_iv is None:
            return None
        return self.put_25_delta_iv - self.call_25_delta_iv


@dataclass
class VolSurface:
    asset: str
    points: list[VolSurfacePoint]
    timestamp: datetime


@dataclass
class FuturesBasis:
    asset: str
    basis_pct: float
    timestamp: datetime


def _get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{DERIBIT_URL}/{path}", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result")
    except Exception:
        return None


def _nearest_expiry(target_days: int, expiries: list[datetime]) -> datetime | None:
    if not expiries:
        return None
    target = datetime.now(timezone.utc) + timedelta(days=target_days)
    return min(expiries, key=lambda d: abs((d - target).days))


def fetch_vol_surface(asset: str) -> VolSurface | None:
    """Fetch a simplified vol surface from Deribit (ATM + 25d skew)."""
    instruments = _get("public/get_instruments", {"currency": asset, "kind": "option"})
    if not instruments:
        return None

    expiries = sorted({
        datetime.fromtimestamp(int(i["expiration_timestamp"]) / 1000, tz=timezone.utc)
        for i in instruments
    })
    points = []

    for tenor in (30, 60, 90):
        expiry = _nearest_expiry(tenor, expiries)
        if not expiry:
            continue
        expiry_ts = int(expiry.timestamp() * 1000)
        expiry_instruments = [
            i for i in instruments if i["expiration_timestamp"] == expiry_ts
        ]
        if not expiry_instruments:
            continue

        index = _get("public/get_index_price", {"index_name": f"{asset.lower()}_usd"})
        if not index:
            continue
        index_price = float(index.get("index_price", 0))
        if not index_price:
            continue

        # pick strikes closest to index, and 25% OTM calls/puts
        def _closest(instrs, target_strike, option_type):
            filt = [i for i in instrs if i["option_type"] == option_type]
            if not filt:
                return None
            return min(filt, key=lambda i: abs(i["strike"] - target_strike))

        atm = _closest(expiry_instruments, index_price, "call")
        call_25 = _closest(expiry_instruments, index_price * 1.25, "call")
        put_25 = _closest(expiry_instruments, index_price * 0.75, "put")

        def _iv(instrument):
            if not instrument:
                return None
            ticker = _get("public/ticker", {"instrument_name": instrument["instrument_name"]})
            if not ticker:
                return None
            return float(ticker.get("mark_iv", 0)) if ticker.get("mark_iv") else None

        atm_iv = _iv(atm)
        call_iv = _iv(call_25)
        put_iv = _iv(put_25)

        points.append(VolSurfacePoint(
            tenor_days=tenor,
            atm_iv=atm_iv,
            call_25_delta_iv=call_iv,
            put_25_delta_iv=put_iv,
        ))

    if not points:
        return None

    return VolSurface(asset=asset, points=points, timestamp=datetime.now(timezone.utc))


def fetch_futures_basis(asset: str) -> FuturesBasis | None:
    """Fetch perpetual basis vs index from Deribit."""
    ticker = _get("public/ticker", {"instrument_name": f"{asset.upper()}-PERPETUAL"})
    if not ticker:
        return None
    mark = float(ticker.get("mark_price", 0))
    index = float(ticker.get("index_price", 0))
    if not mark or not index:
        return None
    basis_pct = (mark - index) / index * 100
    return FuturesBasis(asset=asset, basis_pct=round(basis_pct, 3), timestamp=datetime.now(timezone.utc))
