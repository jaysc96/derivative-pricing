"""Market data capture: a provider-neutral interface and the archive behind it.

    from marketdata import Store, YFinanceAdapter

    adapter = YFinanceAdapter()
    store = Store("data/quotes.db")
    for expiry in adapter.expiries("SPY")[:4]:
        store.write_snapshot(adapter.option_chain("SPY", expiry))

Three pieces. ``adapter`` defines what a provider must supply and validates
what it returns; ``yfinance_adapter`` implements that against the live source;
``store`` persists raw quotes verbatim so the derived layer can be rebuilt from
them whenever the pricing engine changes.

The archive keeps raw observations and nothing computed. Implied volatility is
derived separately and stamped with an engine version, which is what makes the
whole history re-derivable rather than merely accumulated.
"""

from .adapter import (
    ChainSnapshot,
    MalformedResponse,
    MarketDataAdapter,
    NotSupported,
    ProviderError,
    ProviderUnavailable,
    QuoteRecord,
    RateLimited,
    UnderlyingBar,
)
from .secrets import MissingCredential, get_secret, load_dotenv
from .store import FAILURE_REASONS, SchemaMismatch, Store
from .tracked import TRACKED
from .yfinance_adapter import YFinanceAdapter

__all__ = [
    "FAILURE_REASONS",
    "TRACKED",
    "ChainSnapshot",
    "MalformedResponse",
    "MarketDataAdapter",
    "MissingCredential",
    "NotSupported",
    "ProviderError",
    "ProviderUnavailable",
    "QuoteRecord",
    "RateLimited",
    "SchemaMismatch",
    "Store",
    "UnderlyingBar",
    "YFinanceAdapter",
    "get_secret",
    "load_dotenv",
]
