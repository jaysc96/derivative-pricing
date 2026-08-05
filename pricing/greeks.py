"""Shared option contract base and Greek estimation.

Holds the contract constructor that both exercise styles inherit, the Gaussian
helpers the closed-form Greeks need, and the bump-and-reprice machinery in
``priceOption``.

The estimation is central differences with a bump scaled to each input (KTD5).
It replaces one-sided differences that shared a single epsilon across
volatility and interest rate and bumped maturity by a flat 0.05 years — a fifth
of a six-month contract's remaining life. That cost about 2% on rho and up to
64% on theta; the measurements are in ``docs/evidence/pre-fix-failures.md``.

Central differences cost one extra reprice per Greek and remove the first-order
truncation term. Finite differences are the exception for delta and gamma,
which come from the solver's own grid instead — see ``contracts.spatial_greeks``
for why a reprice bump cannot work once the grid follows spot.
"""

import numpy as np
import scipy.stats

from .contracts import (
    BUMP_RATE,
    BUMP_RATE_FLOOR,
    BUMP_SPOT,
    BUMP_TIME,
    BUMP_TIME_CAP,
    BUMP_VOL,
    BUMP_VOL_FLOOR,
    DEFAULT_FD_DT,
    DEFAULT_FD_NODES,
    MIN_FD_STEPS,
    PriceResult,
    fd_grid,
    interpolate_at,
    spatial_greeks,
)

N = scipy.stats.norm.cdf


def n(t):
    """Standard normal probability density."""
    return np.exp(-(t**2) / 2) / np.sqrt(2 * np.pi)


class Option:
    def __init__(
        self,
        option_type,
        initial_price,
        strike_price,
        interest_rate,
        volatility,
        dividend_rate,
        time,
        method,
        start_date=None,
        end_date=None,
    ) -> None:
        self.option_type = option_type
        self.S0 = initial_price
        self.K = strike_price
        self.r = interest_rate
        self.sig = volatility
        self.y = dividend_rate
        self.T = time
        self.phi = 1
        if option_type == 'put':
            self.phi = -1

        self.method_name = method
        self.method = getattr(self, method)

        # Finite-difference resolution. Extent is derived from the contract, so
        # unlike the other discretizations these have usable defaults and the
        # caller can price without setting anything.
        self.fd_nodes = DEFAULT_FD_NODES
        self.fd_dt = DEFAULT_FD_DT

        # Set during a Greek pass to hold the discretization still across the
        # bumps. See _fd_discretization.
        self._fd_pinned = None

    def setSeedVariables(self, seed, n, dt=None):
        self.n = n
        self.dt = dt
        self.seed = seed

    def setTreeSteps(self, n):
        self.n = n

    def setFDResolution(self, nodes=None, dt=None):
        """Refine the finite-difference discretization.

        Resolution only. The grid's extent comes from the contract — see
        ``contracts.fd_grid`` for why the caller no longer chooses it.
        """
        if nodes is not None:
            self.fd_nodes = nodes
        if dt is not None:
            self.fd_dt = dt

    def _fd_discretization(self):
        """Grid and step count for the finite-difference solvers.

        Both are derived from the contract, which makes them move when a Greek
        bump moves the contract — and the step count is an integer, so it moves
        in jumps. At one year the count is 252, and bumping maturity by a day
        lands on 251 and 253: the two repriced values then differ by a change of
        scheme as well as a change of maturity, and the difference between them
        is not a derivative of anything. It showed up as theta 7% out on
        one-year puts, and only there, because at six months the 200-step floor
        binds and the count happens to stay put.

        So a Greek pass pins the discretization first and every reprice inside
        it uses the same grid and the same number of steps. That is also the
        more defensible quantity to differentiate: the derivative of the priced
        contract, not of the contract plus the mesh it happened to land on.
        """
        if self._fd_pinned is not None:
            return self._fd_pinned
        steps = max(MIN_FD_STEPS, int(round(self.T / self.fd_dt)))
        S, dS = fd_grid(self.S0, self.K, self.sig, self.T, self.fd_nodes)
        return S, dS, steps

    def _central_difference(self, attribute, h):
        """``dV/dx`` by repricing at ``x + h`` and ``x - h``.

        Restores the attribute before returning, so a caller can price the same
        contract twice and get the same answer.
        """
        original = getattr(self, attribute)
        try:
            setattr(self, attribute, original + h)
            up = self.method()
            setattr(self, attribute, original - h)
            down = self.method()
        finally:
            setattr(self, attribute, original)
        return (up - down) / (2 * h)

    def priceOption(self):
        """Price at the contract's spot, with Greeks. Uniform across methods."""
        if self.method_name == 'BSM':
            return self.method()

        h_S = BUMP_SPOT * self.S0
        h_sig = max(BUMP_VOL_FLOOR, BUMP_VOL * self.sig)
        h_r = max(BUMP_RATE_FLOOR, BUMP_RATE * abs(self.r))
        h_T = min(BUMP_TIME * self.T, BUMP_TIME_CAP)

        if self.method_name == 'FD':
            # Spatial differences on the solver's own grid. A reprice bump
            # cannot work here: the grid is derived from spot, so moving spot
            # moves the grid with it. See contracts.spatial_greeks.
            self._fd_pinned = self._fd_discretization()
            grid, S, dS = self._fd_solve()
            V = interpolate_at(S, grid, self.S0)
            delta, gamma = spatial_greeks(S, grid, dS, self.S0)
        else:
            V = self.method()
            original = self.S0
            try:
                self.S0 = original + h_S
                up = self.method()
                self.S0 = original - h_S
                down = self.method()
            finally:
                self.S0 = original
            delta = (up - down) / (2 * h_S)
            gamma = (up - 2 * V + down) / h_S**2

        try:
            # Theta is the derivative with respect to *elapsed* time, so it is
            # the negative of the derivative with respect to time remaining.
            theta = -self._central_difference('T', h_T)
            vega = self._central_difference('sig', h_sig)
            rho = self._central_difference('r', h_r)
        finally:
            self._fd_pinned = None

        return PriceResult(
            price=float(V),
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
        )
