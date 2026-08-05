"""U13: the vectorized pricers agree with the scalar loops they replaced.

The scalar implementations are reproduced here rather than referenced from git
history, because a performance refactor's specification *is* its previous
behaviour and a guard that depends on a commit reachable today is not a guard.
They are the pre-U13 bodies verbatim, minus the defects U6 fixed.

Agreement is asserted as exact equality, not within a tolerance. The
vectorization was written to preserve the arithmetic — powers of ``u`` and
``d`` are kept as the product ``u**(i-k) * d**k`` rather than collapsed to
``u**(i-2k)``, and the discount factor is hoisted rather than refactored — so
every one of these comes out bit-identical. A tolerance here would quietly
permit a rewrite that changed the numbers, which for a refactor whose entire
claim is that it changed nothing would defeat the point.
"""

import numpy as np
import pytest

from pricing import American_Option, European_Option

K, R = 100.0, 0.05
STEPS = 120

GRID = [
    (kind, m, T, sig, y)
    for kind in ("call", "put")
    for m in (0.7, 1.0, 1.3)
    for T in (0.25, 2.0)
    for sig in (0.15, 0.45)
    for y in (0.0, 0.05)
]


def scalar_binomial(opt, american):
    """The pre-U13 body: rebuilds the lattice per step, loops per node."""
    n, dt = opt.n, opt.T / opt.n
    u = np.exp(opt.sig * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((opt.r - opt.y) * dt) - d) / (u - d)

    ST = np.array([opt.S0 * u ** (n - i) * d**i for i in range(n + 1)])
    V = np.maximum(opt.phi * (ST - opt.K), 0)

    for i in range(n - 1, -1, -1):
        Vt = np.zeros(i + 1)
        if american:
            St = np.array([opt.S0 * u ** (i - k) * d**k for k in range(i + 1)])
            ev = np.maximum(opt.phi * (St - opt.K), 0)
        for j in range(i + 1):
            cont = np.exp(-opt.r * dt) * (p * V[j] + (1 - p) * V[j + 1])
            Vt[j] = max(ev[j], cont) if american else cont
        V = Vt
    return V[0]


def scalar_trinomial(opt, american):
    n, dt = opt.n, opt.T / opt.n
    u = np.exp(opt.sig * np.sqrt(3 * dt))
    d = 1 / u
    dXu = opt.sig * np.sqrt(3 * dt)

    gam = opt.r - opt.y - opt.sig**2 / 2
    pd = 0.5 * ((opt.sig**2 * dt + gam**2 * dt**2) / dXu**2 - gam * dt / dXu)
    pu = 0.5 * ((opt.sig**2 * dt + gam**2 * dt**2) / dXu**2 + gam * dt / dXu)
    pm = 1 - pd - pu

    ST = np.array(
        [opt.S0 * u ** max(n - i, 0) * d ** max(i - n, 0) for i in range(2 * n + 1)]
    )
    V = np.maximum(opt.phi * (ST - opt.K), 0)

    for i in range(n - 1, -1, -1):
        Vt = np.zeros(2 * i + 1)
        if american:
            St = np.array(
                [opt.S0 * u ** max(i - k, 0) * d ** max(k - i, 0) for k in range(2 * i + 1)]
            )
            ev = np.maximum(opt.phi * (St - opt.K), 0)
        for j in range(2 * i + 1):
            cont = np.exp(-opt.r * dt) * (pu * V[j] + pm * V[j + 1] + pd * V[j + 2])
            Vt[j] = max(ev[j], cont) if american else cont
        V = Vt
    return V[0]


@pytest.mark.parametrize("kind,m,T,sig,y", GRID)
@pytest.mark.parametrize(
    "style,method,reference",
    [
        ("european", "BT", scalar_binomial),
        ("european", "TT", scalar_trinomial),
        ("american", "BT", scalar_binomial),
        ("american", "TT", scalar_trinomial),
    ],
)
def test_vectorized_tree_matches_the_scalar_loop(style, method, reference, kind, m, T, sig, y):
    cls = European_Option if style == "european" else American_Option
    opt = cls(kind, m * K, K, R, sig, y, T, method)
    opt.setTreeSteps(STEPS)

    assert float(opt.method()) == reference(opt, american=style == "american")


@pytest.mark.parametrize("kind,m,T,sig,y", GRID)
def test_monte_carlo_matches_the_path_matrix_it_replaced(kind, m, T, sig, y):
    """Evolving two vectors instead of storing every path changes no number.

    The saving is memory, not arithmetic: at 50,000 paths and 252 steps the old
    version allocated 200MB of path history to read one column of it.
    """
    opt = European_Option(kind, m * K, K, R, sig, y, T, "MC")
    opt.setSeedVariables(42, 4000, T / 40)

    steps = int(opt.T / opt.dt)
    np.random.seed(opt.seed)
    z = np.random.normal(size=(opt.n, steps))
    mu = opt.r - opt.y - opt.sig**2 / 2

    ST1 = np.zeros((opt.n, steps + 1))
    ST2 = np.zeros((opt.n, steps + 1))
    ST1[:, 0] += opt.S0
    ST2[:, 0] += opt.S0
    for i in range(steps):
        ST1[:, i + 1] = ST1[:, i] * np.exp(mu * opt.dt + opt.sig * np.sqrt(opt.dt) * z[:, i])
        ST2[:, i + 1] = ST2[:, i] * np.exp(mu * opt.dt - opt.sig * np.sqrt(opt.dt) * z[:, i])

    if kind == "call":
        payoff = (
            np.where(ST1[:, -1] < opt.K, 0, ST1[:, -1] - opt.K)
            + np.where(ST2[:, -1] < opt.K, 0, ST2[:, -1] - opt.K)
        ) / 2
    else:
        payoff = (
            np.where(ST1[:, -1] > opt.K, 0, opt.K - ST1[:, -1])
            + np.where(ST2[:, -1] > opt.K, 0, opt.K - ST2[:, -1])
        ) / 2

    assert float(opt.MC()) == np.exp(-opt.r * opt.T) * payoff.mean()
