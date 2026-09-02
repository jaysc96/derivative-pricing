"""American-exercise pricing methods.

Four methods: binomial and trinomial trees with an early-exercise test at each
node, Longstaff-Schwartz least-squares Monte Carlo, and Crank-Nicolson finite
differences.

Two defects were fixed here in U6.

``LSMC`` fitted its continuation value by solving the normal equations, and the
4x4 matrix is rank-deficient whenever fewer than four paths are in the money.
Deep out-of-the-money contracts raised ``LinAlgError`` rather than degrading —
48 of the convergence suite's failures, and the only defect in the codebase
that failed loudly. It now uses least squares on the design matrix.

``FD`` tested early exercise against the previous time slice rather than
against the exercise payoff. Unlike the others this cost nothing: the terminal
slice is the payoff and a vanilla American option is worth weakly more the
longer it runs, so the two rules agreed at every node of every time step —
verified by running both inside the solver, zero divergence over 199 steps. It
was fixed because it stops being true the moment value is not monotonic in
maturity, which a discrete dividend across an ex-date would do.

Both tree methods still rebuild the stock lattice inside the backward loop,
which U13 hoists.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .contracts import interpolate_at
from .greeks import Option


class American_Option(Option):
    def BT(self):
        dt = self.T / self.n

        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.y) * dt) - d) / (u - d)

        disc = np.exp(-self.r * dt)

        # The lattice was rebuilt inside the backward loop, which is O(n^2)
        # exponentiations for a set of values that never changes. Powers of u
        # and d are computed once and sliced per step.
        u_pow = u ** np.arange(self.n + 1)
        d_pow = d ** np.arange(self.n + 1)

        V = np.maximum(self.phi * (self.S0 * u_pow[::-1] * d_pow - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            St = self.S0 * u_pow[i::-1] * d_pow[:i + 1]
            ev = np.maximum(self.phi * (St - self.K), 0)
            V = np.maximum(ev, disc * (p * V[:-1] + (1 - p) * V[1:]))
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

        disc = np.exp(-self.r * dt)

        u_pow = u ** np.arange(self.n + 1)
        d_pow = d ** np.arange(self.n + 1)

        ST = self.S0 * np.concatenate([u_pow[:0:-1], d_pow])
        V = np.maximum(self.phi * (ST - self.K), 0)

        for i in range(self.n - 1, -1, -1):
            St = self.S0 * np.concatenate([u_pow[i:0:-1], d_pow[:i + 1]])
            ev = np.maximum(self.phi * (St - self.K), 0)
            V = np.maximum(ev, disc * (pu * V[:-2] + pm * V[1:-1] + pd * V[2:]))
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
            if ITM.size == 0:
                # Nothing to decide: no path can exercise here.
                continue
            Y = (Index[ITM, i + 1:] * EV[ITM, i + 1:] * np.exp(-self.r * np.arange(1, m - i) * dt)).sum(axis=1)

            # Least squares on the design matrix rather than np.linalg.solve on
            # the normal equations. Two reasons. The 4x4 matrix f @ f.T is
            # singular whenever fewer than four paths are in the money, which
            # raised LinAlgError on exactly the deep out-of-the-money contracts
            # the volatility surface depends on. And forming the normal
            # equations squares the condition number, which a basis of powers
            # of the same variable can ill afford. lstsq degrades to the
            # minimum-norm solution instead of failing, so a step with too few
            # paths contributes a weak continuation estimate rather than
            # destroying the valuation.
            f = np.array([St[ITM, i]**j for j in range(4)])
            a, *_ = np.linalg.lstsq(f.T, Y, rcond=None)
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
        S, dS, M = self._fd_discretization()
        dt = self.T / M
        alpha = 0.5

        N_grid = len(S)
        j = S / dS

        exercise = np.maximum(self.phi * (S - self.K), 0)

        CV = np.zeros((N_grid, M))
        CV[:, -1] = exercise

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
            CV[:, i] = lu_solve(lu, np.dot(RA, CV[:, i + 1]) + B)
            # Against the exercise payoff, which is what the American condition
            # actually says. This previously compared against the previous time
            # slice and happened to give identical numbers — the terminal slice
            # is the payoff and vanilla American value never falls as maturity
            # lengthens, so the two agreed at every node. It stops agreeing the
            # moment that monotonicity does, which a discrete dividend would do.
            CV[:, i] = np.maximum(CV[:, i], exercise)

        return CV[:, 0], S, dS
