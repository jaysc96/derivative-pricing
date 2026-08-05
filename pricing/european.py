"""European-exercise pricing methods.

Five methods: closed-form Black-Scholes-Merton, binomial and trinomial trees,
Monte Carlo with antithetic variates, and Crank-Nicolson finite differences.

Two known defects live here, preserved so that U3's extraction stayed provable
against ``tests/baseline_prices.json``:

* ``BSM`` computes vega as ``S * exp(-rT) * n(d2)``, mixing the two equivalent
  vega identities. The result is the true vega scaled by ``S / K`` — correct
  only at the money.
* ``MC`` omits the dividend yield from its drift, so it prices a
  non-dividend-paying asset regardless of the ``y`` it was given.

U6 fixes both.

``FD`` is the one method U4 changed: it builds its own grid from the contract
and returns a scalar at spot rather than a vector across bounds the caller
guessed. ``tests/test_contract.py`` holds it to what the old grid produced at
the same point.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .contracts import MIN_FD_STEPS, PriceResult, fd_grid, interpolate_at
from .greeks import N, Option, n


class European_Option(Option):
    def BSM(self):
        zp = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) + self.sig * np.sqrt(self.T) / 2
        zm = (np.log(self.S0 / self.K) + (self.r - self.y) * self.T) / (self.sig * np.sqrt(self.T)) - self.sig * np.sqrt(self.T) / 2

        V = self.phi * (self.S0 * np.exp(- self.y * self.T) * N(self.phi * zp) - self.K * np.exp(- self.r * self.T) * N(self.phi * zm))

        delta = self.phi * np.exp(- self.y * self.T) * N(self.phi * zp)
        rho = self.phi * self.K * self.T * np.exp(- self.r * self.T) * N(self.phi * zm)
        gamma = np.exp(- self.y * self.T) * n(zp) / self.sig / np.sqrt(self.T) / self.S0
        deltaK = - self.phi * np.exp(- self.r * self.T) * N(self.phi * zm)
        theta = self.r * self.K * deltaK + self.y * self.S0 * delta - self.sig**2 * self.S0**2 * gamma / 2
        vega = self.S0 * np.sqrt(self.T) * np.exp(- self.r * self.T) * n(zm)
        return PriceResult(price=V, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    def MC(self):
        m = int(self.T / self.dt)
        np.random.seed(self.seed)
        z = np.random.normal(size = (self.n, m))
        mu = self.r - (self.sig**2) / 2

        ST1 = np.zeros((self.n, m+1))
        ST2 = np.zeros((self.n, m+1))

        ST1[:,0] += self.S0
        ST2[:,0] += self.S0

        for i in range(m):
            ST1[:, i + 1] = ST1[:, i] * np.exp(mu * self.dt + self.sig * np.sqrt(self.dt) * z[:, i])
            ST2[:, i + 1] = ST2[:, i] * np.exp(mu * self.dt - self.sig * np.sqrt(self.dt) * z[:, i])

        if self.option_type == 'call':
            VTa = (np.where(ST1[:, -1] < self.K, 0, ST1[:, -1] - self.K) + np.where(ST2[:, -1] < self.K, 0, ST2[:, -1] - self.K)) / 2
        else:
            VTa = (np.where(ST1[:, -1] > self.K, 0, self.K - ST1[:, -1]) + np.where(ST2[:, -1] > self.K, 0, self.K - ST2[:, -1])) / 2

        Vta_est = np.exp(-self.r * self.T) * VTa.mean()
        return Vta_est

    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])

        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(i+1)
            for j in range(i+1):
                Vt[j] = np.exp(-self.r * dt) * (p * V[j] + (1-p) * V[j+1])
            V = Vt
        return V[0]

    def TT(self):
        dt = self.T / self.n
        u = np.exp(self.sig * np.sqrt(3 * dt))
        d = 1 / u
        dXu = self.sig * np.sqrt(3*dt)

        gam = self.r - self.y - self.sig**2 / 2
        pd = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 - gam * dt / dXu)
        pu = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 + gam * dt / dXu)
        pm = 1 - pd - pu

        ST = np.array([self.S0 * u**max(self.n - i, 0) * d**max(i - self.n, 0) for i in range(2 * self.n + 1)])
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                Vt[j] = np.exp(-self.r * dt) * (pu * V[j] + pm * V[j+1] + pd * V[j+2])
            V = Vt
        return V[0]

    def FD(self):
        """Price at the contract's spot, interpolated off the internal grid."""
        V, S, _ = self._fd_solve()
        return interpolate_at(S, V, self.S0)

    def _fd_solve(self):
        """Crank-Nicolson backward through time. Returns the grid solution."""
        # A floor, not a step size: short-dated contracts need more steps than
        # a fixed dt gives them. See MIN_FD_STEPS.
        M = max(MIN_FD_STEPS, int(round(self.T / self.fd_dt)))
        dt = self.T / M
        alpha = 0.5

        S, dS = fd_grid(self.S0, self.K, self.sig, self.T, self.fd_nodes)
        N_grid = len(S)
        j = S / dS

        V = np.zeros((N_grid, M))
        V[:, -1] = np.maximum(self.phi * (S - self.K), 0)

        a1 = (self.sig**2 * j**2 + (self.r - self.y) * j) * (1 - alpha) * dt / 2
        a2 = - 1 - (self.sig**2 * j**2 + self.r) * (1 - alpha) * dt
        a3 = (self.sig**2 * j**2 - (self.r - self.y) * j) * (1 - alpha) * dt / 2

        b1 = - ((self.r - self.y) * j + self.sig**2 * j**2) * alpha * dt / 2
        b2 = (self.sig**2 * j**2 + self.r) * alpha * dt - 1
        b3 = ((self.r - self.y) * j - self.sig**2 * j**2) * alpha * dt / 2

        RA = np.zeros((N_grid, N_grid))
        LA = np.zeros((N_grid, N_grid))

        LA[0, 0] = self.phi
        LA[0, 1] = -self.phi
        LA[-1, -1] = -self.phi
        LA[-1, -2] = self.phi

        for i in range(1, N_grid - 1):
            LA[i, i - 1] = a1[i]
            LA[i, i] = a2[i]
            LA[i, i + 1] = a3[i]

            RA[i, i - 1] = b1[i]
            RA[i, i] = b2[i]
            RA[i, i + 1] = b3[i]

        B = np.zeros(N_grid)
        if self.phi == 1:
            B[0] = S[0] - S[1]
        else:
            B[-1] = S[-2] - S[-1]

        # LA does not change across the time loop, so factorize once and reuse
        # it. Same LAPACK path as np.linalg.solve (getrf then getrs), but the
        # O(n^3) factorization happens once instead of M times.
        lu = lu_factor(LA)

        for i in range(M-2,-1,-1):
            V[:, i] = lu_solve(lu, np.dot(RA, V[:, i + 1]) + B)

        return V[:, 0], S, dS
