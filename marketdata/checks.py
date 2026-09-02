"""No-arbitrage quality gate (KTD9's neighbor for correctness rather than
provenance): filter incoming quotes against three model-free bounds before
they reach the solver, recording every violation rather than discarding it
silently (R16, R17).

**A data-quality gate, not a trading signal.** Free-tier quotes cannot
support a tradeable-edge product — apparent violations here are dominated by
stale quotes, bid-ask spread, borrow cost, discrete dividends, and the
early-exercise premium itself, not real arbitrage. What these bounds are
genuinely good for: protecting the volatility surface from quotes that are
internally inconsistent, and evidence the developer understands them.

**Every bound is evaluated on the sides that would actually execute the
trade it describes, never on the mid (R32).** A mid-price check can miss a
real violation the spread would still let through, and can flag a spread
artifact that was never tradeable. Each function's docstring below states
which side plays which role and why.

**The band's textbook lower bound is S - D - K**, D the present value of
discrete dividends over the option's life. This archive tracks a continuous
dividend yield (KTD9's snapshot carries ``dividend_yield``, not a discrete
schedule), so D is restated as its continuous-yield equivalent
``S * (1 - e^(-yT))`` — the standard translation, and the same inequality,
not an approximation of a different one (R32).

**The recency marker is a trade-recency proxy, not a quote-recency one**
(R32): the provider carries a last-trade time, so a quote sitting untouched
for days can carry bid/ask fields populated right now around a stale price.
A leg outside the snapshot window is excluded before any bound evaluates it,
so a stale trade never gets blamed on the bound that caught it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

from .adapter import ChainSnapshot

PUT_CALL_BAND = "put_call_band"
BUTTERFLY = "butterfly"
CALENDAR = "calendar"

#: How far a leg's last trade may sit behind the snapshot's own day before
#: it is excluded rather than checked. A trading day, not an hour: options on
#: the tracked set are liquid but not every contract trades every day, and a
#: window shorter than a day would exclude routine gaps between trades on
#: thin strikes, not just genuinely abandoned quotes.
DEFAULT_STALENESS_WINDOW = timedelta(days=1)


@dataclass(frozen=True)
class Violation:
    """One bound, breached by one leg or leg-group.

    ``legs`` are the involved contracts' symbols, in the order the bound's
    own docstring names them — so a reader (or R21's violations table) knows
    which side of the trade each one played, not just that something failed.
    """

    kind: str
    detail: str
    legs: tuple[str, ...] = ()


def _two_sided(bid: float | None, ask: float | None) -> bool:
    return bid is not None and ask is not None and bid > 0 and ask > 0


def is_stale(
    last_trade_at: datetime | None,
    as_of: date,
    *,
    window: timedelta = DEFAULT_STALENESS_WINDOW,
) -> bool:
    """Whether a leg's last-trade marker falls outside the snapshot's window (R32).

    A missing marker is treated as stale rather than fresh — the absence of
    evidence a contract traded recently is not evidence it did.
    """
    if last_trade_at is None:
        return True
    snapshot_end = datetime.combine(as_of, time.max, tzinfo=timezone.utc)
    return snapshot_end - last_trade_at > window


def check_put_call_band(
    *,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    legs: tuple[str, ...] = (),
) -> Violation | None:
    """R16's American put-call inequality band.

        S*e^(-yT) - K  <=  C - P  <=  S - K*e^(-rT)

    The lower bound is breached only if buying the call at its ask and
    selling the put at its bid still profits against it
    (``call_ask - put_bid < lower``) — that is the actual cost of entering
    the trade that exploits a low band violation (long call, short put,
    short stock, lend K+D), so only a violation surviving real execution
    prices counts. The upper bound mirrors it: selling the call at its bid
    and buying the put at its ask (``call_bid - put_ask > upper``) is what
    exploits a high band violation (short call, long put, long stock,
    borrow K). Returns ``None`` when either leg has no two-sided quote —
    there is nothing tradeable to check.
    """
    if not (_two_sided(call_bid, call_ask) and _two_sided(put_bid, put_ask)):
        return None

    lower = S * math.exp(-y * T) - K
    upper = S - K * math.exp(-r * T)

    if call_ask - put_bid < lower:
        return Violation(
            PUT_CALL_BAND,
            f"call_ask({call_ask:g}) - put_bid({put_bid:g}) = "
            f"{call_ask - put_bid:.4f} < lower bound {lower:.4f}",
            legs,
        )
    if call_bid - put_ask > upper:
        return Violation(
            PUT_CALL_BAND,
            f"call_bid({call_bid:g}) - put_ask({put_ask:g}) = "
            f"{call_bid - put_ask:.4f} > upper bound {upper:.4f}",
            legs,
        )
    return None


def check_butterfly(
    *,
    low: tuple[float, float, float],
    mid: tuple[float, float, float],
    high: tuple[float, float, float],
    legs: tuple[str, ...] = (),
) -> Violation | None:
    """R16's butterfly convexity: price is convex in strike, same type and expiry.

    ``low``, ``mid``, ``high`` are ``(strike, bid, ask)`` triples with
    strictly increasing strikes. Convexity requires the middle strike's price
    sit at or below the chord between the two wings:

        mid  <=  w_low * low + w_high * high
        w_low  = (high.strike - mid.strike) / (high.strike - low.strike)
        w_high = (mid.strike  - low.strike) / (high.strike - low.strike)

    the general form of the familiar evenly-spaced ``low - 2*mid + high >= 0``
    — a real captured chain is not evenly spaced, so the weighted form is
    what actually applies to it. The trade that exploits a violation buys the
    wings (``w_low`` units of low, ``w_high`` units of high, both at ask) and
    sells the body (mid, at bid); only a net credit at those prices is a real
    violation: ``w_low*low_ask + w_high*high_ask - mid_bid < 0``.
    """
    low_strike, low_bid, low_ask = low
    mid_strike, mid_bid, mid_ask = mid
    high_strike, high_bid, high_ask = high

    if not (low_strike < mid_strike < high_strike):
        raise ValueError("low, mid, high must be strictly increasing strikes")
    if not (
        _two_sided(low_bid, low_ask)
        and _two_sided(mid_bid, mid_ask)
        and _two_sided(high_bid, high_ask)
    ):
        return None

    spread = high_strike - low_strike
    w_low = (high_strike - mid_strike) / spread
    w_high = (mid_strike - low_strike) / spread
    cost = w_low * low_ask + w_high * high_ask - mid_bid

    if cost < 0:
        return Violation(
            BUTTERFLY,
            f"wing cost {cost:.4f} < 0 at strikes "
            f"{low_strike:g}/{mid_strike:g}/{high_strike:g}",
            legs,
        )
    return None


def check_calendar(
    *,
    near: tuple[float, float],
    far: tuple[float, float],
    legs: tuple[str, ...] = (),
) -> Violation | None:
    """R16's calendar monotonicity: same strike and type, more time is worth at least as much.

    True for American options regardless of dividends — the longer-dated
    holder can always exercise at the same moment the shorter-dated one
    would and replicate its payoff, so it can never be worth less. ``near``
    and ``far`` are ``(bid, ask)`` pairs, near's expiry earlier than far's.
    The exploiting trade sells near at its bid and buys far at its ask; only
    a net credit today is a real violation (``near_bid > far_ask``).
    """
    near_bid, near_ask = near
    far_bid, far_ask = far

    if not (_two_sided(near_bid, near_ask) and _two_sided(far_bid, far_ask)):
        return None

    if near_bid > far_ask:
        return Violation(
            CALENDAR,
            f"near bid({near_bid:g}) > far ask({far_ask:g})",
            legs,
        )
    return None


def _time_to_expiry(expiry: date, as_of: date) -> float:
    """Years between two dates — the same convention store.py's own julianday query uses."""
    return (expiry - as_of).days / 365.0


def find_violations(
    chains: Sequence[ChainSnapshot],
    *,
    staleness_window: timedelta = DEFAULT_STALENESS_WINDOW,
) -> list[Violation]:
    """Scan every expiry captured for one symbol at one moment against all three bounds.

    ``chains`` should be the same symbol's snapshots from one capture round —
    comparing an expiry captured this morning against one from last week
    would be exactly the point-in-time discipline the archive exists to keep
    (marketdata.store's own docstring); this function trusts the caller for
    that rather than re-deriving it from timestamps.

    Butterfly and calendar checks run on strike- and expiry-adjacent triples
    and pairs only, not every combination. A convexity or monotonicity
    violation on a chain that is locally consistent everywhere else must show
    up between some pair of neighbors — checking every neighbor pair once is
    the standard practical scope, not a coverage shortcut.
    """
    violations: list[Violation] = []
    fresh_by_expiry: dict[date, tuple[ChainSnapshot, list]] = {}

    for chain in chains:
        fresh = [
            q
            for q in chain.quotes
            if not is_stale(q.last_trade_at, chain.as_of, window=staleness_window)
        ]
        fresh_by_expiry[chain.expiry] = (chain, fresh)

    for chain, quotes in fresh_by_expiry.values():
        by_strike_type = {(q.strike, q.option_type): q for q in quotes}
        strikes = sorted({q.strike for q in quotes})

        if (
            chain.underlying_price is not None
            and chain.risk_free_rate is not None
            and chain.dividend_yield is not None
        ):
            T = _time_to_expiry(chain.expiry, chain.as_of)
            for strike in strikes:
                call = by_strike_type.get((strike, "call"))
                put = by_strike_type.get((strike, "put"))
                if call is None or put is None:
                    continue
                violation = check_put_call_band(
                    call_bid=call.bid,
                    call_ask=call.ask,
                    put_bid=put.bid,
                    put_ask=put.ask,
                    S=chain.underlying_price,
                    K=strike,
                    r=chain.risk_free_rate,
                    y=chain.dividend_yield,
                    T=T,
                    legs=(call.contract_symbol, put.contract_symbol),
                )
                if violation:
                    violations.append(violation)

        for option_type in ("call", "put"):
            side_strikes = [s for s in strikes if (s, option_type) in by_strike_type]
            for low_k, mid_k, high_k in zip(side_strikes, side_strikes[1:], side_strikes[2:]):
                low = by_strike_type[(low_k, option_type)]
                mid = by_strike_type[(mid_k, option_type)]
                high = by_strike_type[(high_k, option_type)]
                violation = check_butterfly(
                    low=(low.strike, low.bid, low.ask),
                    mid=(mid.strike, mid.bid, mid.ask),
                    high=(high.strike, high.bid, high.ask),
                    legs=(low.contract_symbol, mid.contract_symbol, high.contract_symbol),
                )
                if violation:
                    violations.append(violation)

    sorted_expiries = sorted(fresh_by_expiry)
    for near_expiry, far_expiry in zip(sorted_expiries, sorted_expiries[1:]):
        _, near_quotes = fresh_by_expiry[near_expiry]
        _, far_quotes = fresh_by_expiry[far_expiry]
        near_by = {(q.strike, q.option_type): q for q in near_quotes}
        far_by = {(q.strike, q.option_type): q for q in far_quotes}
        for key, near_q in near_by.items():
            far_q = far_by.get(key)
            if far_q is None:
                continue
            violation = check_calendar(
                near=(near_q.bid, near_q.ask),
                far=(far_q.bid, far_q.ask),
                legs=(near_q.contract_symbol, far_q.contract_symbol),
            )
            if violation:
                violations.append(violation)

    return violations
