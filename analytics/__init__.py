"""Analytics computed from the archive: the surface and the realized series (R19, R20).

Both read the derived layer and the underlying-bar history — neither calls
the solver, the same discipline KTD9's analytics reads already keep
elsewhere. See ``surface`` for skew and term structure, ``realized`` for
realized volatility.
"""

from .realized import realized_volatility, realized_volatility_series
from .surface import INSUFFICIENT_DATA, SkewCurve, TermStructureCurve, build_skew, build_term_structure

__all__ = [
    "INSUFFICIENT_DATA",
    "SkewCurve",
    "TermStructureCurve",
    "build_skew",
    "build_term_structure",
    "realized_volatility",
    "realized_volatility_series",
]
