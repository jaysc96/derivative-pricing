"""R4: finite-difference Greeks against their closed-form counterparts.

Per Greek, because the failures are per Greek. Delta and gamma already agree to
better than a percent — they come from spatial differences on the solver's own
grid (U4). Theta, vega and rho come from bumping and re-pricing, and all three
are wrong for separate reasons:

* **vega** is not a finite-difference failure at all. ``BSM`` mixes the two
  equivalent identities, computing ``S * exp(-rT) * n(d2)`` where the ``S`` and
  the discount factor belong to different forms. The result is the true vega
  scaled by ``S / K``, so the closed form — not the bumped estimate — is the
  side that is wrong. The ratio in the failure output makes that legible: at
  moneyness 0.8 the bumped vega is 1.25x the closed form, and 1/0.8 = 1.25.
* **theta** uses a one-sided difference with a 0.05-year bump, which is a fifth
  of a six-month contract's remaining life.
* **rho** reuses the volatility epsilon as the interest-rate bump, and is also
  one-sided.

Tolerances are what a correct implementation should meet, not what this one
does. Three of the five fail until U6 rebuilds the estimation on central
differences with per-input bump scales (KTD5).
"""

import pytest

from pricing import European_Option

K, R, SIG, DIVIDEND = 100.0, 0.05, 0.25, 0.02
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.5, 1.0]

# (relative tolerance, absolute floor). The floor matters where a Greek passes
# through zero and a relative test would demand impossible precision — but it
# has to stay small enough not to swallow the defect. An earlier floor of 0.05
# on theta hid a 64% error, because the contract where the one-sided bump does
# most damage is also one where theta is only -0.08. These are what a central
# difference with a per-input bump scale should comfortably achieve; the units
# are the Greek's own, so theta's floor is 0.02 of value per year.
TOLERANCE = {
    "delta": (0.02, 0.005),
    "gamma": (0.03, 0.0005),
    "theta": (0.05, 0.02),
    "vega": (0.02, 0.05),
    "rho": (0.01, 0.02),
}

CASES = [
    (m, T, kind) for m in MONEYNESS for T in MATURITIES for kind in ("call", "put")
]


def both(m, T, kind):
    args = (kind, m * K, K, R, SIG, DIVIDEND, T)
    return (
        European_Option(*args, "FD").priceOption(),
        European_Option(*args, "BSM").priceOption(),
    )


@pytest.mark.parametrize("greek", sorted(TOLERANCE))
@pytest.mark.parametrize("m,T,kind", CASES)
def test_finite_difference_greek_matches_closed_form(greek, m, T, kind):
    fd, closed_form = both(m, T, kind)
    estimated, exact = getattr(fd, greek), getattr(closed_form, greek)

    rel, floor = TOLERANCE[greek]
    limit = max(floor, rel * abs(exact))
    ratio = estimated / exact if exact else float("nan")

    assert abs(estimated - exact) <= limit, (
        f"{greek} at moneyness {m}, T={T}, {kind}: "
        f"finite difference {estimated:.5f} vs closed form {exact:.5f} "
        f"(ratio {ratio:.4f}, allowed {limit:.5f})"
    )


def test_vega_disagreement_tracks_moneyness_exactly():
    """Name the defect rather than only detecting it.

    If the closed form is the true vega scaled by ``S / K``, then the ratio of
    bumped to closed-form vega is exactly ``K / S`` at every moneyness. Pinning
    the shape means U6 cannot make this pass by loosening a tolerance — only by
    fixing the identity.
    """
    for m in MONEYNESS:
        fd, closed_form = both(m, 1.0, "call")
        assert fd.vega / closed_form.vega == pytest.approx(1 / m, rel=0.02), (
            f"at moneyness {m} the ratio is {fd.vega / closed_form.vega:.4f}, "
            f"expected {1 / m:.4f}"
        )
