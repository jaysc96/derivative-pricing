"""American-exercise pricing methods.

Four methods: binomial and trinomial trees with an early-exercise test at each
node, Longstaff-Schwartz least-squares Monte Carlo, and Crank-Nicolson finite
differences.

Two known defects live here, preserved so that U3's extraction stayed provable
against ``tests/baseline_prices.json``:

* ``FD`` tests early exercise against the previous time slice
  (``np.maximum(CV[:, i], CV[:, i + 1])``) rather than against the exercise
  payoff, so it does not enforce the American condition it is meant to.
* ``LSMC`` solves its regression with ``np.linalg.solve`` on a 4x4 normal-equations
  matrix built from the in-the-money paths, which is rank-deficient whenever
  fewer than four paths are in the money. Deep out-of-the-money contracts raise
  ``LinAlgError`` rather than degrading.

Both tree methods also rebuild the stock lattice inside the backward loop, which
U13 hoists. U6 fixes the defects.

``FD`` is the one method U4 changed: it builds its own grid from the contract
and returns a scalar at spot rather than a vector across bounds the caller
guessed. The early-exercise defect above is untouched by that change and still
shows up as American puts pricing above their European counterparts for the
wrong reason.
"""

import numpy as np

from .contracts import MIN_FD_STEPS, fd_grid, interpolate_at
from .greeks import Option


class American_Option(Option):
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        ST = np.array([self.S0 * u**(self.n - i) * d**i for i in range(self.n + 1)])

        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(i+1)
            St = np.array([self.S0 * u**(i - k) * d**k for k in range(i+1)])
            ev = np.maximum(self.phi * (St - self.K),0)
            for j in range(i+1):
                Vt[j] = max(ev[j], np.exp(- self.r * dt) * (p * V[j] + (1-p) * V[j+1]))
            V = Vt
        return V[0]

    def TT(self):
        dt = self.T / self.n
        u = np.exp(self.sig * np.sqrt(3 * dt))
        d = 1 / u
        dXu = self.sig * np.sqrt(3 * dt)

        gam = self.r - self.y - self.sig**2 / 2
        pd = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 - gam * dt / dXu)
        pu = 0.5 * ((self.sig**2 * dt + gam**2 * dt**2) / dXu**2 + gam * dt / dXu)
        pm = 1 - pd - pu

        ST = np.array([self.S0 * u**max(self.n - i, 0) * d**max(i - self.n, 0) for i in range(2 * self.n + 1)])
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            Vt = np.zeros(2 * i + 1)
            St = np.array([self.S0 * u**max(i - k, 0) * d**max(k - i, 0) for k in range(2 * i + 1)])
            ev = np.maximum(self.phi * (St - self.K), 0)
            for j in range(2 * i + 1):
                Vt[j] = max(ev[j], np.exp(-self.r * dt) * (pu * V[j] + pm * V[j+1] + pd * V[j+2]))
            V = Vt
        return V[0]

    def LSMC(self):
        m = int(np.sqrt(self.n))
        dt = self.T / m
        mu = self.r - self.y - self.sig**2 / 2

        np.random.seed(self.seed)
        Z = np.random.normal(size=(self.n // 2, m))

        St = np.zeros((self.n, m + 1))
        St[:,0] = self.S0

        Index = np.zeros((self.n, m))

        for i in range(m):
            St[:self.n // 2, i + 1] = St[:self.n // 2, i] * np.exp(mu * dt + self.sig * np.sqrt(dt) * Z[:, i])
            St[self.n // 2:, i + 1] = St[self.n // 2:, i] * np.exp(mu * dt - self.sig * np.sqrt(dt) * Z[:, i])
        St = St[:, 1:]
        St = St / self.K

        EV = np.zeros((self.n, m))
        EV[:, -1] = np.maximum(self.phi * (St[:, -1] - 1), 0)
        Index[:, -1] = np.where(EV[:, -1] > 0, 1, 0)

        for i in range(m - 2, -1, -1):
            EV[:, i] = np.maximum(self.phi * (St[:, i] - 1), 0)

            ITM = np.where(EV[:, i] > 0)[0]
            Y = (Index[ITM, i + 1:] * EV[ITM, i + 1:] * np.exp(-self.r * np.arange(1, m - i) * dt)).sum(axis=1)

            f = np.array([St[ITM, i]**j for j in range(4)])
            A = np.dot(f, f.T)
            b = np.dot(f, Y)
            a = np.linalg.solve(A, b)
            ECV = np.dot(f.T, a)

            Index[ITM[EV[ITM, i] >= ECV], i] = 1
            Index[ITM[EV[ITM, i] >= ECV], i + 1:] = 0

        V0 = (Index * EV * np.exp(-self.r * np.arange(1, m + 1) * dt)).sum(axis=1)
        return self.K * V0.mean()

    def FD(self):
        """Price at the contract's spot, interpolated off the internal grid."""
        CV, S, _ = self._fd_solve()
        return interpolate_at(S, CV, self.S0)

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

        CV = np.zeros((N_grid, M))
        CV[:, -1] = np.maximum(self.phi * (S - self.K), 0)

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

        for i in range(M-2,-1,-1):
            CV[:, i] = np.linalg.solve(LA, np.dot(RA, CV[:, i + 1]) + B)
            CV[:, i] = np.maximum(CV[:, i], CV[:, i + 1])

        return CV[:, 0], S, dS
