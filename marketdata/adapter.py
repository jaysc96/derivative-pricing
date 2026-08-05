"""The provider-neutral boundary.

Nothing provider-shaped crosses this line — no DataFrame, no Yahoo field name,
no library exception (KTD8). That is what makes a second provider a one-file
change rather than a refactor, and it is the only reason the fallback named in
the plan can be built under time pressure instead of designed under it.

The interface supplies three things: the expiries available for a symbol, an
option chain for a symbol and expiry, and a daily underlying series. The chain
method takes an optional ``as_of`` date — a provider that can answer for a past
date implements it, one that cannot raises ``NotSupported``. That parameter is
here now rather than later because the schema freezes with it: a backfilled row
carries an as-of date distinct from when it was captured, and adding that
afterwards is a migration on the one table that must not be disturbed.

Validation happens at this edge, deliberately. The live provider is an
unofficial scraper, so a renamed field or an unexpected null arrives as
plausible-looking data rather than as an error. Rejected here it is a bad
record; admitted here it is a bad chart three units downstream. This is a
different check from the arbitrage gate later on, which judges whether prices
relate to each other sensibly rather than whether a record is well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

# Beyond these a record is not implausible, it is broken. Deliberately wide —
# the job here is to catch a parse failure or a renamed field, not to have an
# opinion about what a sensible option costs.
MAX_PRICE = 1_000_000.0
MAX_STRIKE = 1_000_000.0
MAX_IMPLIED_VOL = 100.0  # 10,000%, which no real quote reaches


class ProviderError(Exception):
    """Base for every failure that crosses the adapter boundary."""


class RateLimited(ProviderError):
    """The provider refused because we asked too often."""


class ProviderUnavailable(ProviderError):
    """Network failure, timeout, or an HTTP error that is not a rate limit."""


class MalformedResponse(ProviderError):
    """The response arrived but did not have the shape the adapter expects."""


class NotSupported(ProviderError):
    """A capability this provider does not have — a dated chain, typically."""


@dataclass(frozen=True)
class QuoteRecord:
    """One option contract as observed once.

    ``provider_iv`` is the provider's own implied volatility, kept because it
    is free (already in the response) and because the difference between it and
    ours is a product, not a diagnostic.

    ``last_trade_at`` is the only recency marker the live provider returns, and
    it times the last *trade*, not the last quote — a contract quoted all day
    but untraded since Friday carries Friday's stamp beside a current bid and
    ask. It is a liquidity signal. Do not read it as freshness.
    """

    contract_symbol: str
    option_type: str
    strike: float
    bid: float | None
    ask: float | None
    last: float | None
    volume: int | None
    open_interest: int | None
    provider_iv: float | None
    last_trade_at: datetime | None

    @property
    def two_sided(self) -> bool:
        """Both sides quoted, which is the gate a derived value needs.

        The spike found ``ask`` populated on 100% of 3,681 contracts, so in
        practice this is a question about the bid alone — but the derived layer
        needs both sides regardless of which one tends to be missing.
        """
        return bool(self.bid and self.bid > 0 and self.ask and self.ask > 0)

    @property
    def mid(self) -> float | None:
        """Mid of bid and ask, or None when either side is absent."""
        if not self.two_sided:
            return None
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class ChainSnapshot:
    """Every quote for one symbol and expiry, as seen at one moment.

    ``captured_at`` is when we asked. ``as_of`` is the date the quotes describe
    — equal to the capture date for a live pull, earlier for a backfilled one.
    ``origin`` records which, because "the quote we observed" and "the quote a
    vendor records for a past date" are different claims and a series that
    silently mixes them is not auditable.
    """

    provider: str
    symbol: str
    expiry: date
    captured_at: datetime
    as_of: date
    origin: str  # "live" | "backfill"
    underlying_price: float | None
    quotes: tuple[QuoteRecord, ...]


@dataclass(frozen=True)
class UnderlyingBar:
    """One daily bar for the underlying."""

    symbol: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int | None


class MarketDataAdapter(Protocol):
    """What every provider implementation must offer. Three methods (KTD8)."""

    name: str

    def expiries(self, symbol: str) -> tuple[date, ...]:
        """Expiry dates the provider lists for this symbol."""
        ...

    def option_chain(
        self, symbol: str, expiry: date, as_of: date | None = None
    ) -> ChainSnapshot:
        """The chain for one symbol and expiry.

        ``as_of=None`` means now. A date means that date; providers without
        historical chains raise ``NotSupported``.
        """
        ...

    def underlying_history(
        self, symbol: str, start: date, end: date
    ) -> tuple[UnderlyingBar, ...]:
        """Daily bars for the underlying, inclusive of both ends."""
        ...


# --------------------------------------------------------------------------
# Edge validation
# --------------------------------------------------------------------------


def clean_price(value, *, field: str, allow_zero: bool = True) -> float | None:
    """Coerce a provider price, or reject the record.

    Returns None for a genuinely absent value — a missing bid is ordinary, and
    the spike measured it on 31% of contracts. Raises for a value that is
    present but impossible, because that is a parse failure wearing a number's
    clothes.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MalformedResponse(f"{field} is not numeric") from exc

    if number != number:  # NaN, which pandas produces for a missing cell
        return None
    if number < 0:
        raise MalformedResponse(f"{field} is negative: {number}")
    if not allow_zero and number == 0:
        return None
    if number > MAX_PRICE:
        raise MalformedResponse(f"{field} exceeds {MAX_PRICE}: {number}")
    return number


def clean_count(value, *, field: str) -> int | None:
    """Coerce a volume or open-interest count, or reject the record."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MalformedResponse(f"{field} is not numeric") from exc
    if number != number:
        return None
    if number < 0:
        raise MalformedResponse(f"{field} is negative: {number}")
    return int(number)


def clean_implied_vol(value, *, field: str = "provider_iv") -> float | None:
    """Coerce the provider's own implied volatility.

    Zero becomes None rather than an error: providers emit 0.0 to mean "we
    could not compute one", and storing that as a volatility would be a lie
    the derived layer might later believe.
    """
    number = clean_price(value, field=field)
    if number is None or number == 0:
        return None
    if number > MAX_IMPLIED_VOL:
        raise MalformedResponse(f"{field} exceeds {MAX_IMPLIED_VOL}: {number}")
    return number


def validate_strike(value, *, field: str = "strike") -> float:
    """A strike is required, positive, and finite. No record survives without one."""
    number = clean_price(value, field=field)
    if number is None or number <= 0:
        raise MalformedResponse(f"{field} is missing or non-positive: {value!r}")
    if number > MAX_STRIKE:
        raise MalformedResponse(f"{field} exceeds {MAX_STRIKE}: {number}")
    return number


def validate_option_type(value) -> str:
    """Option type is one of two strings, normalized."""
    text = str(value).strip().lower()
    if text not in ("call", "put"):
        raise MalformedResponse(f"option_type is neither call nor put: {value!r}")
    return text
