"""Shared option contract base and Greek estimation.

Holds the contract constructor that both exercise styles inherit, the Gaussian
helpers the closed-form Greeks need, and the bump-and-reprice machinery in
``priceOption``.

Known defects are preserved deliberately so that U3's extraction stayed
provable against ``tests/baseline_prices.json``. The bump machinery here is one
of them: the differences are one-sided, the time bump is 0.05 years regardless
of maturity, and the interest-rate bump reuses the volatility epsilon. U6
rebuilds it on central differences with a bump scaled to each input.
"""

import numpy as np
import scipy.stats

from .contracts import (
    DEFAULT_FD_DT,
    DEFAULT_FD_NODES,
    PriceResult,
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

    def priceOption(self, eps=1):
        """Price at the contract's spot, with Greeks. Uniform across methods."""
        if self.method_name == 'BSM':
            return self.method()

        time_eps = 0.05
        sig_eps = eps / 200

        if self.method_name == 'FD':
            grid, S, dS = self._fd_solve()
            V = interpolate_at(S, grid, self.S0)
            delta, gamma = spatial_greeks(S, grid, dS, self.S0)
        else:
            V = self.method()
            self.S0 += eps
            Vp = self.method()
            self.S0 -= 2 * eps
            Vm = self.method()
            self.S0 += eps

            delta = (Vp - Vm) / 2 / eps
            gamma = (Vp + Vm - 2 * V) / (eps**2)

        self.T += time_eps
        VTp = self.method()
        self.T -= time_eps
        theta = (V - VTp) / time_eps

        self.sig += sig_eps
        VSigp = self.method()
        self.sig -= sig_eps
        vega = (VSigp - V) / sig_eps

        self.r += sig_eps
        Vrp = self.method()
        self.r -= sig_eps
        rho = (Vrp - V) / sig_eps

        return PriceResult(
            price=float(V),
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
        )
