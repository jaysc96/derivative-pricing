"""The shared result type, and the grid the finite-difference solvers build.

Every method returns a :class:`PriceResult` — a scalar price at the contract's
spot plus its Greeks — so a caller pricing a chain never has to ask which
method produced a number before it can read it.

That uniformity is what forces the grid in here. The finite-difference solvers
are the only methods with a spatial discretization, and they used to take its
bounds from the caller and hand back a vector across it, never reading the
contract's own spot at all. A caller who guessed the bounds badly got a
plausible-looking curve computed on a grid that did not contain the contract:
the web form defaulted to 90 and 110, so an at-the-money option was priced with
spot one cell from the boundary. The grid is now derived from the contract and
the price interpolated back to spot, leaving the caller with only the
resolution knob.
"""

import math
from dataclasses import dataclass

import numpy as np

#: Nodes across the spatial grid. Resolution, not extent — the caller may raise
#: it for accuracy but cannot move the boundary away from the contract.
DEFAULT_FD_NODES = 400

#: One trading day. The Crank-Nicolson scheme is unconditionally stable, so this
#: bounds accuracy rather than convergence.
DEFAULT_FD_DT = 1.0 / 252.0

#: Floor on the number of time steps, whatever ``DEFAULT_FD_DT`` works out to.
#:
#: A step per trading day sounds contract-derived but is not: it gives a
#: three-year option 756 steps and a one-month option 21, when the short-dated
#: contract is the one that needs more. Crank-Nicolson loses its second-order
#: accuracy against the kink in the payoff, so the error decays like ``1/M``
#: rather than ``1/M^2``, and near expiry the kink dominates the whole
#: solution. Measured on a one-month contract, 21 steps misprices by 2.8% at
#: every volatility from 0.1 to 1.2 — and refining the *space* grid instead
#: makes it worse (2.84% -> 3.55% from 400 to 1600 nodes), which is the
#: signature of the oscillation rather than of a coarse grid. 200 steps brings
#: that to roughly 0.3% and costs about a fifth of a second.
MIN_FD_STEPS = 200


@dataclass(frozen=True)
class PriceResult:
    """A price at the contract's spot and the five Greeks that go with it.

    Unrounded. Rounding is a presentation decision and belongs at the display
    edge — convergence tests and implied-volatility inversion both need more
    precision than a price is ever shown with.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


#: Grid extent, as a multiple of ``max(S0, K)``. The lower bound keeps the
#: boundary clear of the contract when volatility or maturity is near zero. The
#: upper bound is the awkward one — see ``fd_grid``.
MIN_SPAN, MAX_SPAN = 2.5, 12.0


def fd_grid(S0, K, sig, T, nodes=DEFAULT_FD_NODES):
    """Spatial grid for a contract, descending from ``S_max`` to ``dS``.

    The extent has to hold the terminal distribution, which argues for scaling
    it as ``exp(k * sig * sqrt(T))``. But this grid is uniform in ``S`` while
    that width is log-normal, so the two pull against each other: a wide enough
    boundary for high volatility and long maturity spreads the same node budget
    so thinly that spot lands between distant nodes. At ``sig=0.8, T=3`` an
    uncapped ``exp(5 * sig * sqrt(T))`` puts the boundary at 1400x strike with a
    node every 357 currency units, which prices a 140 call about 1140 too high.

    So the span is clamped. Truncating the grid costs accuracy at the boundary,
    but far above the strike the value is nearly linear in ``S`` and the Neumann
    condition carries it, whereas a grid too coarse to resolve spot has nothing
    to fall back on. Measured against closed form over spot 60-140, ``sig`` up
    to 0.6 and ``T`` up to 2 years, the error is at worst 0.11 and typically
    0.02. Beyond that envelope truncation starts to dominate — the worst case
    at ``sig=0.8, T=3`` is about 0.4 — because the boundary assumes delta
    reaches 1 while a dividend-paying call approaches ``exp(-y * T)``. A grid
    uniform in ``log(S)`` would dissolve the tension; that is a change to the
    scheme, not to this function.

    The coefficients use ``j = S / dS`` as the node index counted from ``S = 0``,
    so the grid must be uniform with spacing ``dS`` and must start one step above
    zero. Choosing ``dS = S_max / nodes`` and running down to ``dS`` satisfies
    both exactly: node ``i`` sits at ``(nodes - i) * dS``.

    Returns the grid and its spacing.
    """
    anchor = max(S0, K)
    span = min(max(math.exp(4.0 * sig * math.sqrt(T)), MIN_SPAN), MAX_SPAN)
    S_max = anchor * span
    dS = S_max / nodes
    return np.linspace(S_max, dS, nodes), dS


def interpolate_at(S, values, S0):
    """Read a grid quantity at spot. ``S`` descends, so flip for ``np.interp``."""
    return float(np.interp(S0, S[::-1], values[::-1]))


def spatial_greeks(S, V, dS, S0):
    """Delta and gamma from differences across the solver's own grid.

    Not from re-pricing. Once the grid is derived from the contract rather than
    supplied, a re-priced central difference has nothing left to bump — moving
    spot moves the grid with it, and a bump smaller than one cell returns the
    same interpolated price twice, so gamma comes back as exactly zero. The
    solver already knows the value at every neighbouring node, which is the
    same information a bump was trying to recover.
    """
    Sa, Va = S[::-1], V[::-1]

    delta = np.empty_like(Va)
    gamma = np.empty_like(Va)

    delta[1:-1] = (Va[2:] - Va[:-2]) / (2 * dS)
    gamma[1:-1] = (Va[2:] - 2 * Va[1:-1] + Va[:-2]) / dS**2

    # One-sided at the two boundary nodes. Spot is far inside the grid by
    # construction, so these are never the values that get interpolated.
    delta[0], delta[-1] = delta[1], delta[-2]
    gamma[0], gamma[-1] = gamma[1], gamma[-2]

    return (
        float(np.interp(S0, Sa, delta)),
        float(np.interp(S0, Sa, gamma)),
    )
