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
    first = rows[0]
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


def _latest_moment(store: Store) -> datetime | None:
    last_capture = store.coverage()["last_capture"]
    return datetime.fromisoformat(last_capture) if last_capture else None


def build_skew(
    store: Store,
    symbol: str,
    *,
    engine_version: int = ENGINE_VERSION,
    option_type: str = "call",
    min_points: int = MIN_SKEW_POINTS,
) -> list[SkewCurve]:
    """One skew curve per captured expiry, from that expiry's latest snapshot."""
    moment = _latest_moment(store)
    if moment is None:
        return []

    expiries = sorted({e for s, e in store.symbols_and_expiries() if s == symbol})
    excluded = _violating_legs(store, symbol, expiries, moment)

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
) -> list[TermStructureCurve]:
    """One term-structure curve per strike that appears in the latest snapshots, across expiries."""
    moment = _latest_moment(store)
    if moment is None:
        return []

    expiries = sorted({e for s, e in store.symbols_and_expiries() if s == symbol})
    excluded = _violating_legs(store, symbol, expiries, moment)

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
