# Pre-fix suite failures

What the convergence, Greek, and structural suites reported **before** any
defect was fixed. U5 and U6 land on one branch so continuous integration never
observes this state, which means without this file the repository would hold a
green suite arriving beside its own fixes and no evidence the tests ever caught
anything.

Captured at `2873e3a`, on Python 3.12.5, numpy 2.5.1, scipy 1.18.0. Raw output
alongside this file in `pre-fix-pytest-output.txt`.

```
$ pytest tests/test_convergence.py tests/test_greeks.py tests/test_structural.py
152 failed, 1145 passed in 219.91s
```

## The failures partition by defect

Every failure maps to exactly one cause, and no cause produces failures outside
its own group. That is the property worth having: a suite where failures smear
across unrelated tests cannot tell you what to fix.

| Count | Test | Cause |
|---:|---|---|
| 80 | `test_european_methods_agree_with_dividends` | Monte Carlo drift omits the dividend yield |
| 48 | `test_american_methods_agree_*` | Longstaff-Schwartz regression matrix goes singular |
| 16 | `test_finite_difference_greek_matches_closed_form[…vega]` | Closed-form vega mixes the two identities |
| 7 | `test_finite_difference_greek_matches_closed_form[…rho]` | Rho bump reuses the volatility epsilon |
| 1 | `test_finite_difference_greek_matches_closed_form[…theta]` | One-sided difference over a 0.05-year bump |
| 0 | `test_structural.py` (all) | — see "What did not fail" |

### Monte Carlo dividend drift — 80 failures

Exactly the four pairs that include `MC`, twenty grid points each:

```
20 BSM-MC     20 BT-MC     20 TT-MC     20 MC-FD
```

Every pair *not* involving Monte Carlo passes on the same grid, and all
Monte Carlo pairs pass on the zero-dividend grid
(`test_european_methods_agree_without_dividends`, 0 failures). That split is the
diagnosis: the sampling, the antithetic variates, and the discounting are all
sound — only the drift is wrong. It reads `mu = r - sig^2/2` where its five
sibling methods read `r - y - sig^2/2`, so it prices a non-payer whatever `y`
it is handed.

Worst deviation 2.94 on a contract worth 27.4, or 10.7%, at moneyness 1.2 and
one year.

### Longstaff-Schwartz singular matrix — 48 failures

All three pairs that include `LSMC`, sixteen grid points each. These are not
tolerance failures; they are crashes:

```
numpy.linalg.LinAlgError: Singular matrix
```

`LSMC` fits a cubic by normal equations, `A = f @ f.T` with `f` a 4×N basis over
the in-the-money paths. When fewer than four paths are in the money `A` is rank
deficient and `np.linalg.solve` raises. It fires on exactly the deep
out-of-the-money contracts — moneyness 0.8 and 0.9 calls, 1.1 and 1.2 puts —
which is the region the volatility surface most needs, and the region where
inversion brackets downward.

This is the fourth defect. It was found while capturing the U3 baseline, is not
in R3 as written, and needs the requirement amended.

### Vega identity — 16 failures

Failures at every moneyness except 1.0, which is the signature. The ratio of
bumped to closed-form vega:

| Moneyness | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|
| ratio | 1.2588 | 1.1103 | passes | 0.9096 | 0.8392 |
| 1/moneyness | 1.2500 | 1.1111 | 1.0000 | 0.9091 | 0.8333 |

The closed form computes `S * exp(-rT) * n(d2)`, taking the spot term from one
vega identity and the discount factor and density from the other. The result is
the true vega scaled by `S / K` — correct only at the money, and wrong by 25%
one strike out. The finite-difference estimate is the correct side here.

`test_vega_disagreement_tracks_moneyness_exactly` asserts that shape directly,
so the fix cannot be a loosened tolerance.

### Greek machinery — 8 failures

Seven rho, one theta. Both come from `priceOption`: the differences are
one-sided, the time bump is a flat 0.05 years regardless of maturity, and the
interest-rate bump reuses the volatility epsilon. Rho lands about 2% out, theta
up to 64% where theta itself is near zero.

Smaller than the other three, and genuinely so — this is the defect whose
practical cost the suite showed to be lowest, not highest.

## What did not fail

**Every structural test passed** — put-call parity, American at or above
European, American at or above immediate exercise, Merton's no-early-exercise
result for calls on non-payers, price bounds, and monotonicity in spot. 336
assertions, no failures.

The plan expected these to fail, on the grounds that the American
finite-difference solver tests early exercise against the previous time slice
(`np.maximum(CV[:, i], CV[:, i + 1])`) rather than against the exercise payoff.
**That expectation was wrong, and the tests are right to pass.**

The terminal slice *is* the payoff, and a vanilla American option is worth
weakly more the longer it has to run, so `V(tau_{i+1}) >= payoff` holds at every
node by induction. That makes the two rules the same number everywhere: in the
exercise region both equal the payoff, in the continuation region both equal the
continuation value. Running both rules side by side inside the solver over a
two-year American put confirms it — **zero divergence at any node of any of the
199 time steps**, and prices identical to every digit:

```
previous-slice rule   14.49833561
exercise-payoff rule  14.49833561
```

Against a converged independent binomial the American finite-difference solver
is within 0.03 absolute and 0.81% across a forty-contract sweep, with no value
below its payoff and none below its European counterpart.

So the third item in R3 is real as code and inert as arithmetic. It still wants
fixing — it states a condition it does not mean, and is correct only by
accident. The moment value stops being monotonic in maturity, which discrete
dividends across an ex-date would do, it silently starts returning wrong
answers. But it is a robustness fix, not a pricing fix, and U6 should be
described that way.

## Consequences for the plan

1. **R3 needs a fourth defect.** The Longstaff-Schwartz singular matrix is the
   most severe of the four — it is the only one that fails loudly rather than
   quietly, and it fails in the region the surface work depends on.
2. **R3's third item should be reclassified** from a pricing defect to a
   robustness one, with the evidence above.
3. **The defect severity order is not what the plan assumed.** By measured
   impact: Longstaff-Schwartz crash, then Monte Carlo drift (10.7%), then vega
   (25% off the money), then the bump machinery (2%), then the American
   early-exercise condition (0%).
