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


def fd_grid(S0, K, sig, T, nodes=DEFAULT_FD_NODES):
    """Spatial grid for a contract, descending from ``S_max`` to ``dS``.

    The extent has to hold the terminal distribution: five log-normal standard
    deviations above the larger of spot and strike, floored at three times it so
    that a near-zero volatility or a near-dated expiry still leaves the
    boundary far enough away for the Neumann conditions to hold.

    The scheme's coefficients use ``j = S / dS`` as the node index counted from
    ``S = 0``, so the grid must be uniform with spacing ``dS`` and must start one
    step above zero. Choosing ``dS = S_max / nodes`` and running down to ``dS``
    satisfies both exactly: node ``i`` sits at ``(nodes - i) * dS``.

    Returns the grid and its spacing.
    """
    anchor = max(S0, K)
    span = max(3.0, math.exp(5.0 * sig * math.sqrt(T)))
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
