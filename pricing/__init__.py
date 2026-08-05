"""Option pricing by six independent methods.

Two contract classes, split by exercise style::

    from pricing import European_Option, American_Option

    opt = European_Option("call", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0, "BSM")
    result = opt.priceOption()

Each class is constructed with the contract and the name of the method that
should price it. The method must be one this exercise style supports —
``EUROPEAN_METHODS`` and ``AMERICAN_METHODS`` enumerate them. Trees and Monte
Carlo need their discretization set before pricing, via ``setTreeSteps`` and
``setSeedVariables``; finite differences derive their grid from the contract and
price without configuration, though ``setFDResolution`` will refine it.

The methods and where they live:

===========  ==========================================  =========  ========
Method       Technique                                   European   American
===========  ==========================================  =========  ========
``BSM``      Black-Scholes-Merton closed form            yes        no
``BT``       Cox-Ross-Rubinstein binomial tree           yes        yes
``TT``       Trinomial tree                              yes        yes
``MC``       Monte Carlo with antithetic variates        yes        no
``LSMC``     Longstaff-Schwartz least-squares MC         no         yes
``FD``       Crank-Nicolson finite differences           yes        yes
===========  ==========================================  =========  ========

``BSM`` returns a scalar price and closed-form Greeks. ``FD`` returns a vector
across its own grid rather than a value at spot. Everything else returns a
scalar price with bumped Greeks. U4 replaces this with one uniform contract.

This package is a behavior-preserving extraction of the former
``src/option.py``. Known defects were carried over deliberately so the move
could be proven against ``tests/baseline_prices.json``; each module docstring
names the ones it holds, and U6 fixes them.
"""

from .american import American_Option
from .contracts import PriceResult
from .european import European_Option
from .greeks import N, Option, n

#: Methods each exercise style can be constructed with.
EUROPEAN_METHODS = ("BSM", "BT", "TT", "MC", "FD")
AMERICAN_METHODS = ("BT", "TT", "LSMC", "FD")

__all__ = [
    "AMERICAN_METHODS",
    "EUROPEAN_METHODS",
    "American_Option",
    "European_Option",
    "N",
    "Option",
    "PriceResult",
    "n",
]
