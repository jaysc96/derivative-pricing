"""The contracts every suite prices.

Spread across moneyness, maturity, exercise style, and dividend. The spread is
the point: three of this codebase's four known defects are invisible at the
money with no dividend, so a table that only exercised the obvious case would
agree with a broken implementation.
"""

from pricing import American_Option, European_Option

#  label:              (option_type,  S0,    K,     r,    sig,  y,    T)
CASES = {
    "atm_call_1y": ("call", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "atm_put_1y": ("put", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "itm_call_1y": ("call", 120.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "otm_call_1y": ("call", 80.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "itm_put_6m": ("put", 80.0, 100.0, 0.03, 0.35, 0.00, 0.5),
    "otm_put_2y": ("put", 130.0, 100.0, 0.05, 0.25, 0.04, 2.0),
    "high_vol_call_3m": ("call", 100.0, 100.0, 0.01, 0.60, 0.00, 0.25),
    "low_vol_call_1y": ("call", 100.0, 95.0, 0.05, 0.10, 0.02, 1.0),
}

STYLES = {
    "european": European_Option,
    "american": American_Option,
}


def option(style, label, method):
    """Construct the contract a ``style.label`` pair names, priced by ``method``."""
    return STYLES[style](*CASES[label], method)
