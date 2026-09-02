"""The symbol set the capture job currently tracks -- the single place it is defined.

R37 has an analytics visitor select a symbol from this same set, and the API
re-validates a requested symbol against it before any query runs. Both
``scripts/capture.py`` and ``api/analytics_routes.py`` import ``TRACKED`` from
here rather than each keeping their own copy, which is what keeps the two from
drifting the way U2's own tracked-set test already guards against for the
provider spike.
"""

from __future__ import annotations

#: Provisional, and deliberately the same six the spike measures — a tracked
#: set that drifts from the set being validated makes the spike's numbers
#: describe something other than what is being captured. U2 narrows this to the
#: plan's three-to-six once enough daily observations have accumulated; until
#: then, capturing all six costs one request per symbol more and keeps the
#: choice open, which is the cheaper mistake while history is unrecoverable.
TRACKED = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA")
