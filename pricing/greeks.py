"""Shared option contract base and Greek estimation.

Holds the contract constructor that both exercise styles inherit, the Gaussian
helpers the closed-form Greeks need, and the bump-and-reprice machinery in
``priceOption``.

This module is a faithful move of the base class from the former
``src/option.py``. Known defects are preserved deliberately so that the
extraction can be proven behavior-preserving against ``tests/baseline_prices.json``;
U6 rebuilds the Greek estimation here.
"""

import numpy as np
import scipy.stats

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

    def setSeedVariables(self, seed, n, dt=None):
        self.n = n
        self.dt = dt
        self.seed = seed

    def setTreeSteps(self, n):
        self.n = n

    def setFDVariables(self, S_min, S_max, dt):
        self.S_min = S_min
        self.S_max = S_max
        self.dt = dt

    def priceOption(self, eps=1):
        if self.method_name == 'BSM':
            return self.method()

        time_eps = 0.05
        sig_eps = eps / 200
        S = []

        if self.method_name != 'FD':
            V = self.method()
            self.S0 += eps
            Vp = self.method()
            self.S0 -= 2 * eps
            Vm = self.method()
            self.S0 += eps
        else:
            V, S = self.method(return_S=True)
            self.S_min += eps
            self.S_max += eps
            Vp = self.method()
            self.S_min -= 2 * eps
            self.S_max -= 2 * eps
            Vm = self.method()
            self.S_max += eps
            self.S_min += eps

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

        if len(S) > 0:
            return {
                "stock_price": S,
                "price": np.round(V, 3),
                "delta": np.round(delta, 3),
                "gamma": np.round(gamma, 3),
                "theta": np.round(theta, 3),
                "vega": np.round(vega, 3),
                "rho": np.round(rho, 3),
            }

        return {
            "price": np.round(V, 3),
            "delta": np.round(delta, 3),
            "gamma": np.round(gamma, 3),
            "theta": np.round(theta, 3),
            "vega": np.round(vega, 3),
            "rho": np.round(rho, 3),
        }
