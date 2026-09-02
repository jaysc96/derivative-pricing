"""Anchors from outside this repository, plus the reference used to check them.

Six methods agreeing with each other proves they share assumptions, not that
the assumptions are right — every one of them could be wrong in the same
direction and the convergence suite would stay green. R31 exists to close that
gap, and KTD10 chose textbook values over a second library on the grounds that
matching another implementation only proves agreement with someone else's
possible bug.

Two anchors, both from Hull's *Options, Futures, and Other Derivatives*.

**Citation caveat, stated plainly.** The parameter sets and printed values below
are from Hull, but the edition and page were not verifiable while writing this,
and searching did not surface a quotable copy. What *was* done is stronger than
nothing and weaker than a citation: each value is reproduced by
``independent_crr`` / ``independent_bsm`` in this module, which share no code
with ``pricing``. The European pair reproduces to 4.7594 and 0.8086 against
Hull's printed 4.76 and 0.81; the American illustration reproduces to 4.4885
against a printed 4.49. Agreement to the printed precision from an independent
implementation makes a mis-remembered parameter set unlikely, but it is not a
substitute for checking the book. Confirm against a copy before this is cited
as an external anchor in the README.

One trap worth recording: Hull prints 4.49 for the American put as the result
of a **five-step** tree, and the converged value is 4.2842. Anchoring a
200-step method to 4.49 would have been wrong by 0.2 in the direction of
looking correct. The anchor below therefore pins the five-step tree, which is
what the book actually computed.
"""

import math

import numpy as np
from scipy.stats import norm


def independent_bsm(kind, S, K, r, sig, y, T):
    """Black-Scholes-Merton, written from the formula. Shares nothing with ``pricing``."""
    d1 = (math.log(S / K) + (r - y + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-y * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-y * T) * norm.cdf(-d1)


def independent_crr(kind, S, K, r, sig, y, T, n, american):
    """Vectorized Cox-Ross-Rubinstein tree, written from the recurrence.

    Deliberately a second implementation rather than a call into ``pricing``:
    an anchor computed by the code under test anchors nothing.
    """
    dt = T / n
    u = math.exp(sig * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - y) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    phi = 1 if kind == "call" else -1

    j = np.arange(n + 1)
    V = np.maximum(phi * (S * u ** (n - j) * d**j - K), 0.0)

    for i in range(n - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            k = np.arange(i + 1)
            np.maximum(V, phi * (S * u ** (i - k) * d**k - K), out=V)
    return float(V[0])


def converged_american(kind, S, K, r, sig, y, T, n=4000):
    """Richardson extrapolation over two step counts, to remove the O(1/n) tree bias."""
    coarse = independent_crr(kind, S, K, r, sig, y, T, n, True)
    fine = independent_crr(kind, S, K, r, sig, y, T, 2 * n, True)
    return 2 * fine - coarse


# --------------------------------------------------------------------------
# The anchors
# --------------------------------------------------------------------------

#: Hull's Black-Scholes worked example. Six-month option on a non-dividend
#: payer. Printed to two decimals, which is the precision asserted.
HULL_EUROPEAN = {
    "params": dict(S=42.0, K=40.0, r=0.10, sig=0.20, y=0.0, T=0.5),
    "call": 4.76,
    "put": 0.81,
    "printed_decimals": 2,
    "reproduced": {"call": 4.759422, "put": 0.808599},
}

#: Hull's binomial illustration: a five-month American put, valued on a tree of
#: exactly five steps. The step count is part of the anchor, not an
#: implementation detail — see the module docstring.
HULL_AMERICAN_FIVE_STEP = {
    "params": dict(S=50.0, K=50.0, r=0.10, sig=0.40, y=0.0, T=5 / 12),
    "steps": 5,
    "put": 4.49,
    "printed_decimals": 2,
    "reproduced": {"put": 4.488460},
}

#: What that same contract is actually worth, for the methods that converge.
#: Not a published figure — computed by ``converged_american`` here.
HULL_AMERICAN_CONVERGED = {
    "params": dict(S=50.0, K=50.0, r=0.10, sig=0.40, y=0.0, T=5 / 12),
    "put": 4.28422,
}
