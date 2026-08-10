"""Implied volatility against realized, aligned by day (R20, F3).

Implied here is the front-month mean call implied volatility: the derived
layer's solved values for the nearest expiry still live as of each capture,
averaged into one figure per day. That is a coarser read than the skew and
term-structure curves ``surface.py`` renders -- this module exists to put a
single trend line against realized volatility over accumulated history, not
to reproduce the surface a second way. Realized comes from
``analytics.realized.realized_volatility_series`` over the same symbol's
underlying bars. Either side is ``None`` on a day the other has no value yet,
so a fresh archive shows one series arriving before the other rather than an
empty chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from marketdata import Store
from pricing import ENGINE_VERSION

from .realized import DEFAULT_WINDOW, realized_volatility_series
from .surface import live_expiries


@dataclass(frozen=True)
class ComparisonPoint:
    as_of: date
    implied_vol: float | None
    realized_vol: float | None


def build_implied_vs_realized(
    store: Store,
    symbol: str,
    *,
    engine_version: int = ENGINE_VERSION,
    window: int = DEFAULT_WINDOW,
) -> list[ComparisonPoint]:
    """One point per day either series has a value, sorted ascending."""
    implied_by_date: dict[date, float] = {}
    for as_of, captured_at in store.capture_moments(symbol):
        expiries = live_expiries(store, symbol, captured_at)
        if not expiries:
            continue
        rows = store.derived_as_of(symbol, expiries[0], captured_at, engine_version)
        vols = [row["implied_vol"] for row in rows if row["option_type"] == "call"]
        if vols:
            implied_by_date[as_of] = sum(vols) / len(vols)

    realized_by_date = dict(realized_volatility_series(store.underlying_bars(symbol), window=window))

    dates = sorted(set(implied_by_date) | set(realized_by_date))
    return [
        ComparisonPoint(
            as_of=d, implied_vol=implied_by_date.get(d), realized_vol=realized_by_date.get(d)
        )
        for d in dates
    ]
