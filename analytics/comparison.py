"""Implied volatility against realized, aligned by day (R20, F3).

Implied here is the **at-the-money** call implied volatility of the nearest
expiry that still has a usable quote: the strike closest to the spot the
snapshot itself recorded. Realized comes from
``analytics.realized.realized_volatility_series`` over the same symbol's
underlying bars. Either side is ``None`` on a day the other has no value yet,
so a fresh archive shows one series arriving before the other rather than an
empty chart.

**Why at-the-money rather than a mean across the chain.** This originally
averaged every solved call on the front expiry, which put the wings and the
body on equal footing. Measured on the real archive that read 94.5% against a
realized 9.7-18.2% -- not a divergence, an artifact: the front expiry is
frequently zero or one day out, where deep out-of-the-money implied vols
explode and dominate any unweighted average. The at-the-money strike is the
conventional single-number summary of a surface and the one actually
comparable to a realized figure; the same days read 15.2% and 14.5%.

**Violating legs are excluded, and that matters more here than on the
surface** (R21). ``build_skew`` drops a violating leg from a curve of dozens,
where one bad quote moves little. At the money the chosen leg *is* the whole
answer, so a single leg failing a no-arbitrage bound would set the day's
number outright -- and on the archive as it stands the nearest-spot leg is a
violating one more often than not.

**The front expiry is the nearest one with a clean solved call, not simply
the nearest one.** A zero-days-to-expiry chain routinely has nothing left to
invert, and blindly taking ``expiries[0]`` dropped those capture days from
the series entirely rather than falling through to the expiry behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from marketdata import Store
from pricing import ENGINE_VERSION

from .realized import DEFAULT_WINDOW, realized_volatility_series
from .surface import live_expiries, violating_legs


@dataclass(frozen=True)
class ComparisonPoint:
    as_of: date
    implied_vol: float | None
    realized_vol: float | None


def _atm_implied(store: Store, symbol: str, captured_at, engine_version: int) -> float | None:
    """At-the-money call implied vol for one capture moment, or ``None``.

    ``None`` rather than a fallback figure whenever the inputs are not there
    -- no spot, no live expiry, or nothing left after the quality gate. A day
    with no defensible reading is a gap in the series, not a zero.
    """
    spot = store.underlying_price_at(symbol, captured_at)
    if spot is None:
        return None

    expiries = live_expiries(store, symbol, captured_at)
    if not expiries:
        return None

    excluded = violating_legs(store, symbol, expiries, captured_at)
    for expiry in expiries:
        candidates = [
            row
            for row in store.derived_as_of(symbol, expiry, captured_at, engine_version)
            if row["option_type"] == "call" and row["contract_symbol"] not in excluded
        ]
        if candidates:
            nearest = min(candidates, key=lambda row: abs(row["strike"] - spot))
            return nearest["implied_vol"]
    return None


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
        implied = _atm_implied(store, symbol, captured_at, engine_version)
        if implied is not None:
            implied_by_date[as_of] = implied

    realized_by_date = dict(realized_volatility_series(store.underlying_bars(symbol), window=window))

    dates = sorted(set(implied_by_date) | set(realized_by_date))
    return [
        ComparisonPoint(
            as_of=d, implied_vol=implied_by_date.get(d), realized_vol=realized_by_date.get(d)
        )
        for d in dates
    ]
