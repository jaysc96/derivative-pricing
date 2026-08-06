"""Invert an observed price to an implied volatility, or say why you cannot.

Bracketed root-finding, not Newton (KTD4). American vega has no closed form, so
a derivative-based method would need a bumped vega on every iteration — the
inner loop of an already-slow tree price. Bracketing also hands us the explicit
no-solution signal for free: a price outside the bracket endpoints has no
implied volatility, and the solver says so instead of converging to a boundary
and returning a number that looks like an answer.

**The upper bracket is 500% annualised**, and it is the endpoint that binds in
practice. Single-name out-of-the-money wings routinely imply volatilities above
200%; a ceiling below them would silently reclassify exactly the strikes the
skew view exists to show, and the reclassification would surface downstream as
an insufficient-data state indistinguishable from thin quotes.

**The lower bracket stays above the volatility at which the binomial
risk-neutral probability leaves [0, 1].** With ``u = exp(sig*sqrt(dt))`` and
``d = 1/u``, the probability ``(exp((r-y)*dt) - d) / (u - d)`` is only in range
while ``sig >= |r - y| * sqrt(dt)``. That binds near zero and loosens as the
step count rises, so it is a correctness guard rather than a practical limit —
but without it the tree returns prices from an arbitrage-violating lattice and
the root-finder happily inverts them.

Bracket exhaustion is a distinct outcome from no-solution, because one is a
limit of this solver and the other is a statement about the quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .american import American_Option
from .european import European_Option

#: Annualised. See the module docstring for why it is this high.
MAX_VOL = 5.0

#: Absolute floor, applied on top of the binomial constraint.
MIN_VOL = 1e-4

#: Tree steps for American inversion. Every iteration of the root-finder pays
#: this, so it trades accuracy against the throughput gate.
DEFAULT_STEPS = 200

#: Root-finder tolerance in volatility points — 1e-6 is far below quote
#: granularity and costs a handful of extra iterations.
TOLERANCE = 1e-6

#: Prices this close together are the same price. Both endpoints come from the
#: same pricer, and where exercise dominates both return the payoff exactly, so
#: the comparison is against floating-point noise rather than quote granularity.
PRICE_TOLERANCE = 1e-9

SOLVED = "solved"
NO_SOLUTION = "no_solution"
BRACKET_EXHAUSTED = "bracket_exhausted"
NOT_IDENTIFIED = "not_identified"
NO_QUOTE = "no_quote"


@dataclass(frozen=True)
class InversionResult:
    """What the solver concluded, and enough to tell why."""

    status: str
    implied_vol: float | None = None
    target_price: float | None = None
    price_at_floor: float | None = None
    price_at_ceiling: float | None = None
    detail: str = ""

    @property
    def solved(self) -> bool:
        return self.status == SOLVED


def minimum_volatility(r: float, y: float, T: float, steps: int) -> float:
    """Lowest volatility at which the binomial lattice stays arbitrage-free.

    Below this the risk-neutral probability leaves [0, 1] and the tree prices a
    lattice that admits arbitrage — which the root-finder would invert without
    complaint.
    """
    dt = T / steps
    return max(MIN_VOL, abs(r - y) * math.sqrt(dt) * (1 + 1e-9))


def _price(kind, style, sig, S, K, r, y, T, steps):
    if style == "european":
        return European_Option(kind, S, K, r, sig, y, T, "BSM").BSM().price
    option = American_Option(kind, S, K, r, sig, y, T, "BT")
    option.setTreeSteps(steps)
    return float(option.BT())


def implied_volatility(
    kind: str,
    style: str,
    target_price: float,
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    *,
    steps: int = DEFAULT_STEPS,
) -> InversionResult:
    """Invert one price. Never raises on an unsolvable quote — reports instead.

    ``style`` is ``"european"`` or ``"american"``. ``target_price`` is the mid
    of bid and ask; a quote missing either side should not reach here.
    """
    if target_price is None or target_price <= 0:
        return InversionResult(status=NO_QUOTE, detail="price is absent or non-positive")
    if T <= 0:
        return InversionResult(status=NO_QUOTE, detail="contract has expired")

    floor = minimum_volatility(r, y, T, steps)
    at_floor = _price(kind, style, floor, S, K, r, y, T, steps)
    at_ceiling = _price(kind, style, MAX_VOL, S, K, r, y, T, steps)

    # Below the floor price the quote is cheaper than any admissible volatility
    # can produce — most often a price below intrinsic value, which is a
    # statement about the quote rather than about the solver.
    if target_price < at_floor - PRICE_TOLERANCE:
        return InversionResult(
            status=NO_SOLUTION,
            target_price=target_price,
            price_at_floor=at_floor,
            price_at_ceiling=at_ceiling,
            detail="price below the value at minimum volatility",
        )

    # Sitting *on* the floor price is a different thing, and it is the American
    # case rather than an edge case. A deep in-the-money American put where
    # immediate exercise dominates is worth its payoff at every volatility up
    # to some threshold — the price is flat in sigma, vega is zero, and the
    # quote does not identify a volatility at all. Returning the bracket floor
    # here would hand back 0.0001 for a contract genuinely trading at 12%,
    # which is precisely the misleading value an explicit failure exists to
    # prevent. The information is real and belongs in the surface as a gap.
    if abs(target_price - at_floor) <= PRICE_TOLERANCE:
        return InversionResult(
            status=NOT_IDENTIFIED,
            target_price=target_price,
            price_at_floor=at_floor,
            price_at_ceiling=at_ceiling,
            detail="price is flat in volatility here — at the exercise boundary",
        )

    if target_price > at_ceiling + PRICE_TOLERANCE:
        return InversionResult(
            status=BRACKET_EXHAUSTED,
            target_price=target_price,
            price_at_floor=at_floor,
            price_at_ceiling=at_ceiling,
            detail=f"price exceeds the value at {MAX_VOL:.0%} volatility",
        )

    # Symmetric with the floor test above, and resolved here rather than left
    # to the root-finder. A price a fraction above the ceiling makes the
    # residual negative at *both* ends of the bracket, so Brent refuses it and
    # the failure surfaces as no-solution — which would be a third answer to
    # the same question, and the wrong one.
    if target_price >= at_ceiling - PRICE_TOLERANCE:
        return InversionResult(
            status=SOLVED,
            implied_vol=MAX_VOL,
            target_price=target_price,
            price_at_floor=at_floor,
            price_at_ceiling=at_ceiling,
        )

    def residual(sig):
        return _price(kind, style, sig, S, K, r, y, T, steps) - target_price

    try:
        root = brentq(residual, floor, MAX_VOL, xtol=TOLERANCE, maxiter=200)
    except (ValueError, RuntimeError) as exc:
        # The bracket was checked above, so reaching here means the price
        # function is not monotone across it — worth surfacing rather than
        # papering over.
        return InversionResult(
            status=NO_SOLUTION,
            target_price=target_price,
            price_at_floor=at_floor,
            price_at_ceiling=at_ceiling,
            detail=f"root-finder failed: {type(exc).__name__}",
        )

    return InversionResult(
        status=SOLVED,
        implied_vol=float(root),
        target_price=target_price,
        price_at_floor=at_floor,
        price_at_ceiling=at_ceiling,
    )


def round_trip_error(
    kind: str,
    style: str,
    sig: float,
    S: float,
    K: float,
    r: float,
    y: float,
    T: float,
    *,
    steps: int = DEFAULT_STEPS,
) -> float:
    """Price at ``sig``, invert it, and return the volatility error.

    The measurement R10 is stated in. Kept beside the solver rather than in the
    test file because the evidence generator reports it too.
    """
    price = _price(kind, style, sig, S, K, r, y, T, steps)
    result = implied_volatility(kind, style, price, S, K, r, y, T, steps=steps)
    if not result.solved:
        return float("inf")
    return abs(result.implied_vol - sig)
