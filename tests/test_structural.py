"""R5: relationships that hold by no-arbitrage, whatever the method.

Convergence tests catch a method that disagrees with its siblings. These catch
a whole family agreeing on something impossible — put-call parity and the
American-above-European relation are true of the contracts, not of any
particular way of valuing them, so they fail even when every method is wrong in
the same direction.

**On the American finite-difference early-exercise test.** The plan expected
these to fail until U6, on the grounds that ``FD`` compares the continuation
value against the previous time slice rather than against the exercise payoff.
They pass, and the reason is worth recording because it changes what U6 is for.

The terminal slice *is* the payoff, and a vanilla American option is worth
weakly more the longer it has to run. So ``V(tau_{i+1}) >= payoff`` holds at
every node by induction, which makes ``max(cont, V(tau_{i+1}))`` and
``max(cont, payoff)`` the same number: in the exercise region both equal the
payoff, in the continuation region both equal the continuation value.
Instrumenting the solver to run both rules side by side over a two-year
American put confirms it — zero divergence at any node of any of the 199 time
steps, and identical prices to every digit.

So that defect is real as code and inert as arithmetic. It still wants fixing,
because it states a condition it does not mean and is only correct by accident:
the moment the value stops being monotonic in maturity — discrete dividends
across an ex-date, most obviously — it silently starts returning the wrong
answer. But it is a robustness fix, not a pricing fix, and the suite should not
pretend to catch what it cannot.
"""

import pytest

from pricing import American_Option, European_Option

K, R, SIG = 100.0, 0.05, 0.25
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.5, 2.0]
DIVIDENDS = [0.0, 0.06]
TREE_STEPS = 200

DETERMINISTIC = ["BT", "TT", "FD"]
GRID = [(m, T, y) for m in MONEYNESS for T in MATURITIES for y in DIVIDENDS]


def price(cls, kind, m, y, T, method):
    opt = cls(kind, m * K, K, R, SIG, y, T, method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(TREE_STEPS)
    if method == "BSM":
        return opt.BSM().price
    return float(opt.method())


# --------------------------------------------------------------------------
# Put-call parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["BSM"] + DETERMINISTIC)
@pytest.mark.parametrize("m,T,y", GRID)
def test_european_put_call_parity(method, m, T, y):
    """``C - P = S*exp(-yT) - K*exp(-rT)``, exactly, for any consistent method."""
    import math

    call = price(European_Option, "call", m, y, T, method)
    put = price(European_Option, "put", m, y, T, method)

    forward = m * K * math.exp(-y * T) - K * math.exp(-R * T)
    # Both legs carry the method's own discretization error, so allow two of
    # them. Closed form gets no slack at all beyond floating point.
    slack = 0.0 if method == "BSM" else 0.15
    assert call - put == pytest.approx(forward, abs=slack + 1e-9)


# --------------------------------------------------------------------------
# American against European, and against immediate exercise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", DETERMINISTIC)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("m,T,y", GRID)
def test_american_is_never_worth_less_than_european(method, kind, m, T, y):
    """The American holder can always choose not to exercise early."""
    american = price(American_Option, kind, m, y, T, method)
    european = price(European_Option, kind, m, y, T, method)
    assert american >= european - 1e-6


@pytest.mark.parametrize("method", DETERMINISTIC)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("m,T,y", GRID)
def test_american_is_never_worth_less_than_immediate_exercise(method, kind, m, T, y):
    """Below the payoff there is a riskless profit in buying and exercising."""
    american = price(American_Option, kind, m, y, T, method)
    payoff = max((1 if kind == "call" else -1) * (m * K - K), 0.0)
    assert american >= payoff - 1e-6


@pytest.mark.parametrize("method", DETERMINISTIC)
@pytest.mark.parametrize("m,T", [(m, T) for m in MONEYNESS for T in MATURITIES])
def test_american_call_on_a_non_payer_equals_european(method, m, T):
    """Merton: with no dividend, early exercise of a call is never optimal.

    A sharper test than the inequality above, because it fixes the value from
    both sides. A method that padded American values to satisfy the other two
    tests would fail this one.
    """
    american = price(American_Option, "call", m, 0.0, T, method)
    european = price(European_Option, "call", m, 0.0, T, method)
    assert american == pytest.approx(european, abs=1e-6)


# --------------------------------------------------------------------------
# Bounds that hold for any option at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["BSM"] + DETERMINISTIC)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("m,T,y", GRID)
def test_prices_are_non_negative_and_bounded(method, kind, m, T, y):
    value = price(European_Option, kind, m, y, T, method)
    ceiling = m * K if kind == "call" else K
    assert -1e-6 <= value <= ceiling + 1e-6


@pytest.mark.parametrize("method", ["BSM"] + DETERMINISTIC)
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("T,y", [(T, y) for T in MATURITIES for y in DIVIDENDS])
def test_value_is_monotonic_in_spot(method, kind, T, y):
    """Calls rise with spot, puts fall. Non-monotonicity means an arbitrage."""
    values = [price(European_Option, kind, m, y, T, method) for m in MONEYNESS]
    ordered = values if kind == "call" else values[::-1]
    assert all(a <= b + 1e-6 for a, b in zip(ordered, ordered[1:])), values
