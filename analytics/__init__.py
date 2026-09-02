"""Analytics computed from the archive: the surface, realized vol, and their comparison (R19-R21).

All three read the derived layer and the underlying-bar history — none calls
the solver, the same discipline KTD9's analytics reads already keep
elsewhere. See ``surface`` for skew, term structure, and recorded violations;
``realized`` for realized volatility; ``comparison`` for implied against
realized over time.
"""

from .comparison import ComparisonPoint, build_implied_vs_realized
from .realized import realized_volatility, realized_volatility_series
from .surface import (
    INSUFFICIENT_DATA,
    SkewCurve,
    TermStructureCurve,
    ViolationRow,
    build_skew,
    build_term_structure,
    build_violations,
)

__all__ = [
    "INSUFFICIENT_DATA",
    "ComparisonPoint",
    "SkewCurve",
    "TermStructureCurve",
    "ViolationRow",
    "build_implied_vs_realized",
    "build_skew",
    "build_term_structure",
    "build_violations",
    "realized_volatility",
    "realized_volatility_series",
]
