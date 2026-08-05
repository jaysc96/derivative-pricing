# Pricing benchmarks

Before and after U13's vectorization. Regenerate with:

```
python benchmarks/bench_pricing.py
```

The baseline in `benchmarks/baseline.json` is the post-U6, pre-vectorization
code, recorded on the same machine in the same session. Absolute times depend on
the hardware; the ratio is the claim.

Measured on Python 3.12.5, numpy 2.5.1, scipy 1.18.0. One at-the-money one-year
contract, priced every way it can be — 200 tree steps, 50,000 Monte Carlo paths,
20,000 Longstaff-Schwartz paths, a 400-node finite-difference grid. Each figure
is the median of a full `priceOption` call, which is what a caller actually
pays: one price plus the repricings the Greeks need.

| Method | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| `european.BSM` | 0.26 | 0.25 | 1.0x |
| `european.BT` | 176.02 | 6.74 | **26.1x** |
| `european.TT` | 646.24 | 9.99 | **64.7x** |
| `european.MC` | 6511.65 | 4490.40 | 1.5x |
| `european.FD` | 238.76 | 241.70 | 1.0x |
| `american.BT` | 352.21 | 18.00 | **19.6x** |
| `american.TT` | 943.16 | 22.97 | **41.1x** |
| `american.LSMC` | 6109.95 | 6145.23 | 1.0x |
| `american.FD` | 242.71 | 247.89 | 1.0x |
| **total** | **15220.96** | **11183.16** | **1.4x** |

## Reading the total honestly

**The 1.4x total is the least informative number in the table.** It is dominated
by the two methods this unit did not target: Monte Carlo and Longstaff-Schwartz
were 83% of the baseline total and are 95% of it now. The work was on the four
trees, and there the range is 19.6x to 64.7x.

Nothing about the finite-difference or closed-form rows changed here, and the
1.0x is correct rather than disappointing. The finite-difference gain — taking
the LU factorization of a constant matrix out of a 200-to-756 step loop — landed
during U4, because U5's convergence grid would otherwise have inherited the slow
solver. It is already inside the baseline. Counting it in this ratio would be
claiming it twice.

## What changed

**Both tree families, both exercise styles.** The backward pass iterated node by
node in Python. It is now three array operations per step: the binomial
recurrence becomes `disc * (p * V[:-1] + (1 - p) * V[1:])` and the trinomial one
adds a third slice.

**The American lattice, hoisted.** Both American trees rebuilt the stock lattice
inside the backward loop — `O(n^2)` exponentiations for a set of values that never
changes. Powers of `u` and `d` are now computed once and sliced per step. This is
why the American trees start slower than the European ones and end at a similar
place.

**Monte Carlo, evolved in place.** It allocated two full path matrices and read
one column of each. At 50,000 paths and 252 steps that is 200MB to obtain 400KB
of terminal values. Two vectors now. The arithmetic per step is untouched, which
is why the gain is 1.5x rather than large — the cost was always the 252 Python
iterations over 50,000-element exponentials, and those remain.

## The numbers did not move

Every optimization here is arithmetic-preserving, and that is asserted rather
than hoped: `tests/test_vectorization.py` reproduces the pre-change scalar loops
and checks **exact equality**, not agreement within a tolerance, across 240
contracts spanning moneyness 0.7 to 1.3, two maturities, two volatilities, and
both dividend cases.

That was a constraint on how the vectorization was written, not a lucky outcome.
Powers stay as the product `u**(i-k) * d**k` rather than collapsing to
`u**(i-2k)`, because `d = 1/u` is not exact in floating point and the collapsed
form differs in the last bits. The discount factor is hoisted out of the loop
rather than refactored into the recurrence.

A separate check against the packaged pre-change code over 600 contracts —
including Monte Carlo — also came back bit-identical.

## Where the time is now

Monte Carlo and Longstaff-Schwartz, by a wide margin. Both have a clear next
step, and neither is in U13's scope because both change the method rather than
its implementation:

- **Monte Carlo** simulates 252 steps to read a terminal value. A vanilla
  European payoff depends only on `S_T`, which geometric Brownian motion gives
  exactly in one step. That is a ~250x reduction, but it consumes random numbers
  differently and so changes every recorded Monte Carlo figure — a method
  change, needing its own before-and-after.
- **Longstaff-Schwartz** keeps a full `paths x exercise-dates` indicator matrix
  and re-scans its tail at every step, which is `O(n * m^2)`. The standard
  formulation carries a single cashflow vector instead, at `O(n * m)`. With
  m = 141 that is the dominant term.

Both are worth doing before U12's inversion work multiplies every pricer call by
a root-finder's iteration count.
