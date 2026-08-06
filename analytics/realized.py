"""Realized volatility from stored underlying history (R20).

Annualized sample standard deviation of daily log returns — the standard
convention, and the same 252-trading-day year this project already uses
elsewhere for trading-day-scaled quantities (``pricing.contracts.DEFAULT_FD_DT``
is one trading day, not a coincidence).
"""

from __future__ import annotations

import math
from datetime import date

TRADING_DAYS_PER_YEAR = 252

#: Trailing window for the rolling series, in trading days. Roughly one
#: calendar month — short enough that the figure reacts to a real regime
#: change within weeks, long enough that one wild day does not dominate it.
DEFAULT_WINDOW = 21


def _log_returns(closes: list[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def realized_volatility(closes: list[float]) -> float | None:
    """Annualized realized volatility over one window of daily closes.

    ``None`` below two closes (nothing to compute a return from) or three
    (sample standard deviation needs at least two returns) — never ``0.0``,
    which would read as "no volatility observed" rather than "not enough
    history to say."
    """
    if len(closes) < 3:
        return None
    returns = _log_returns(closes)
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)


def realized_volatility_series(
    bars: list, *, window: int = DEFAULT_WINDOW
) -> list[tuple[date, float]]:
    """Rolling ``window``-day realized volatility, one point per day once enough history exists.

    ``bars`` are oldest-first rows carrying ``bar_date`` and ``close``
    (``Store.underlying_bars``'s own shape). Returns ``(date, value)`` pairs
    keyed on the *last* day of each window, so aligning this against the
    implied-volatility series (R20) is a plain join on date — no
    interpolation or resampling hidden in either series.
    """
    closes = [bar["close"] for bar in bars]
    dates = [date.fromisoformat(bar["bar_date"]) for bar in bars]
    series = []
    for end in range(window, len(closes) + 1):
        window_closes = closes[end - window : end]
        vol = realized_volatility(window_closes)
        if vol is not None:
            series.append((dates[end - 1], vol))
    return series
