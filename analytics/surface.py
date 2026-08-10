"""Skew and term structure from the derived layer (R19), with R36's insufficient-data rule.

Both views read ``Store.derived_as_of`` — solved implied volatilities only,
never the solver — and both exclude any leg implicated in a U16 arbitrage
violation before building a curve (R21): the gate exists precisely so a
violation never quietly reaches an analytics view. Reconstructing a
``ChainSnapshot`` from the flattened store rows lets this module run
``marketdata.checks.find_violations`` itself — the actual arbitrage math —
rather than deciding which legs are excluded a second, possibly divergent
way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from marketdata import ChainSnapshot, QuoteRecord, Store
from marketdata.checks import find_violations
from pricing import ENGINE_VERSION

INSUFFICIENT_DATA = "insufficient_data"
OK = "ok"

#: Fewer than three strikes cannot show a shape — "skew" is a claim about
#: curvature, and two points are just a line.
MIN_SKEW_POINTS = 3

#: Same reasoning across expiries instead of strikes: two points show a
#: slope, not whether the term structure is humped, upward, or downward.
MIN_TERM_STRUCTURE_POINTS = 3


@dataclass(frozen=True)
class SkewPoint:
    strike: float
    implied_vol: float


@dataclass(frozen=True)
class SkewCurve:
    expiry: date
    option_type: str
    status: str
    points: tuple[SkewPoint, ...] = ()


@dataclass(frozen=True)
class TermPoint:
    expiry: date
    implied_vol: float


@dataclass(frozen=True)
class TermStructureCurve:
    strike: float
    option_type: str
    status: str
    points: tuple[TermPoint, ...] = ()


def _reconstruct_chain(store: Store, symbol: str, expiry: date, moment: datetime) -> ChainSnapshot | None:
    """Rebuild a ``ChainSnapshot`` from ``Store.chain_as_of``'s flattened rows.

    ``chain_as_of`` already carries every field a ``QuoteRecord`` and a
    ``ChainSnapshot`` were written from; this folds the flat rows back into
    that shape so ``find_violations`` can run against real archive data
    without a second read path or a second dataclass.
    """
    rows = store.chain_as_of(symbol, expiry, moment)
    if not rows:
        return None
    # The chain's spot/rate/yield come from whichever row happens first in
    # the query's own return order, not from the most recently observed one
    # -- with several quotes at different `observed_at` times in the same
    # chain (the ordinary case once a symbol has been captured more than
    # once), an arbitrary row could carry a stale snapshot's spot/rate/yield
    # alongside a fresher quote's strike/bid/ask.
    first = max(rows, key=lambda row: row["observed_at"])
    quotes = tuple(
        QuoteRecord(
            contract_symbol=row["contract_symbol"],
            option_type=row["option_type"],
            strike=row["strike"],
            bid=row["bid"],
            ask=row["ask"],
            last=row["last"],
            volume=row["volume"],
            open_interest=row["open_interest"],
            provider_iv=row["provider_iv"],
            last_trade_at=(
                datetime.fromisoformat(row["last_trade_at"]) if row["last_trade_at"] else None
            ),
        )
        for row in rows
    )
    return ChainSnapshot(
        provider="stored",
        symbol=symbol,
        expiry=expiry,
        captured_at=datetime.fromisoformat(first["captured_at"]),
        as_of=date.fromisoformat(first["as_of"]),
        origin=first["origin"],
        underlying_price=first["underlying_price"],
        quotes=quotes,
        risk_free_rate=first["risk_free_rate"],
        dividend_yield=first["dividend_yield"],
    )


def _violating_legs(store: Store, symbol: str, expiries: list[date], moment: datetime) -> set[str]:
    chains = [
        chain
        for chain in (_reconstruct_chain(store, symbol, expiry, moment) for expiry in expiries)
        if chain is not None
    ]
    legs: set[str] = set()
    for violation in find_violations(chains):
        legs.update(violation.legs)
    return legs


def _latest_moment(store: Store, symbol: str) -> datetime | None:
    return store.latest_capture(symbol)


def live_expiries(store: Store, symbol: str, moment: datetime) -> list[date]:
    """Expiries not yet settled as of ``moment``, sorted ascending.

    ``symbols_and_expiries`` returns every expiry the archive has ever
    captured, including contracts that expired months ago. Reading those
    into a "current" skew or term-structure curve renders options nobody can
    trade as though they were live, and feeds ``_violating_legs`` chains from
    different capture rounds — a settled leg from one round compared against
    a live leg from another is not the calendar violation it would look
    like. Scoping to ``moment``'s own date also bounds read cost by the
    archive's currently-live surface rather than by its total history.
    """
    return sorted(e for s, e in store.symbols_and_expiries() if s == symbol and e >= moment.date())


def build_skew(
    store: Store,
    symbol: str,
    *,
    engine_version: int = ENGINE_VERSION,
    option_type: str = "call",
    min_points: int = MIN_SKEW_POINTS,
    excluded_legs: set[str] | None = None,
) -> list[SkewCurve]:
    """One skew curve per captured expiry, from that expiry's latest snapshot.

    ``excluded_legs`` lets a caller building both this and
    ``build_term_structure`` for the same symbol compute ``_violating_legs``
    once and pass it to both, rather than each independently reconstructing
    and re-checking the same chains. Computed here when not supplied, so
    calling this alone is unchanged.
    """
    moment = _latest_moment(store, symbol)
    if moment is None:
        return []

    expiries = live_expiries(store, symbol, moment)
    excluded = (
        excluded_legs if excluded_legs is not None else _violating_legs(store, symbol, expiries, moment)
    )

    curves = []
    for expiry in expiries:
        rows = store.derived_as_of(symbol, expiry, moment, engine_version)
        points = sorted(
            (
                SkewPoint(strike=row["strike"], implied_vol=row["implied_vol"])
                for row in rows
                if row["option_type"] == option_type
                and row["contract_symbol"] not in excluded
            ),
            key=lambda p: p.strike,
        )
        status = OK if len(points) >= min_points else INSUFFICIENT_DATA
        curves.append(SkewCurve(expiry=expiry, option_type=option_type, status=status, points=tuple(points)))
    return curves


def build_term_structure(
    store: Store,
    symbol: str,
    *,
    engine_version: int = ENGINE_VERSION,
    option_type: str = "call",
    min_points: int = MIN_TERM_STRUCTURE_POINTS,
    excluded_legs: set[str] | None = None,
) -> list[TermStructureCurve]:
    """One term-structure curve per strike that appears in the latest snapshots, across expiries.

    See ``build_skew`` for ``excluded_legs``.
    """
    moment = _latest_moment(store, symbol)
    if moment is None:
        return []

    expiries = live_expiries(store, symbol, moment)
    excluded = (
        excluded_legs if excluded_legs is not None else _violating_legs(store, symbol, expiries, moment)
    )

    by_strike: dict[float, list[TermPoint]] = {}
    for expiry in expiries:
        rows = store.derived_as_of(symbol, expiry, moment, engine_version)
        for row in rows:
            if row["option_type"] != option_type or row["contract_symbol"] in excluded:
                continue
            by_strike.setdefault(row["strike"], []).append(
                TermPoint(expiry=expiry, implied_vol=row["implied_vol"])
            )

    curves = []
    for strike in sorted(by_strike):
        points = sorted(by_strike[strike], key=lambda p: p.expiry)
        status = OK if len(points) >= min_points else INSUFFICIENT_DATA
        curves.append(
            TermStructureCurve(strike=strike, option_type=option_type, status=status, points=tuple(points))
        )
    return curves


@dataclass(frozen=True)
class ViolationRow:
    """One leg of one recorded violation (R21), flattened for direct display.

    A put-call-band violation names two legs and a butterfly names three;
    each gets its own row here rather than one row per violation, so R21's
    table can show "strike, expiry, contract side" per leg the way the plan
    asks rather than a caller re-deriving that from ``Violation.legs`` itself.
    """

    expiry: date
    strike: float
    option_type: str
    contract_symbol: str
    kind: str
    detail: str


def build_violations(store: Store, symbol: str) -> list[ViolationRow]:
    """Every no-arbitrage violation on the symbol's live surface, one row per leg (R21).

    Reruns the same reconstruction and ``find_violations`` call
    ``_violating_legs`` makes to compute what ``build_skew``/
    ``build_term_structure`` exclude — this is that computation's other
    half, the one that shows what was excluded and why instead of quietly
    dropping it.
    """
    moment = _latest_moment(store, symbol)
    if moment is None:
        return []

    expiries = live_expiries(store, symbol, moment)
    chains = [
        chain
        for chain in (_reconstruct_chain(store, symbol, expiry, moment) for expiry in expiries)
        if chain is not None
    ]
    quotes_by_symbol = {
        quote.contract_symbol: (quote, chain.expiry) for chain in chains for quote in chain.quotes
    }

    rows = []
    for violation in find_violations(chains):
        for leg in violation.legs:
            match = quotes_by_symbol.get(leg)
            if match is None:
                continue
            quote, expiry = match
            rows.append(
                ViolationRow(
                    expiry=expiry,
                    strike=quote.strike,
                    option_type=quote.option_type,
                    contract_symbol=leg,
                    kind=violation.kind,
                    detail=violation.detail,
                )
            )
    return sorted(rows, key=lambda r: (r.expiry, r.strike, r.option_type))
