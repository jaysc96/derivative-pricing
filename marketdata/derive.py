"""The derived layer: implied volatility computed from the raw archive (KTD9).

Raw quotes stay the only source of truth (KTD9, R34). What lives here is a
batch step — invert every raw quote not yet derived at the current engine
version — and a wholesale rebuild that wipes and recomputes when that version
changes. Nothing here writes to the raw layer, and nothing that reads the
derived table calls the solver: an analytics view built on it costs rows
returned, not pricer calls, which is what keeps a page load proportional to
what it displays rather than to the accumulated history behind it.

**Every symbol currently tracked is priced American.** SPY, QQQ, and IWM are
ETF options; AAPL, MSFT, and NVDA are equities. US-listed options on both are
American-style — only cash-settled index options (SPX and similar) are
European, and none is tracked. ``style`` is still an explicit parameter rather
than a hardcoded literal, because the day a European-style symbol joins the
tracked set, a single scalar override stops being correct and a real per-
symbol resolution has to replace it; that day has not arrived.

**A missing rate or yield is a distinct reason from every status the solver
itself produces.** ``pricing.implied``'s four failure statuses describe the
solver's own conclusion about a price; ``NO_MARKET_CONTEXT`` describes a gap
before the solver is ever called — the snapshot the quote belongs to has no
risk-free rate or dividend yield to invert against, most often because a
capture's rate/yield fetch failed while the chain fetch it rode alongside
still succeeded. Both live in the same ``status`` column, because a reader
asking "why isn't this contract in the surface" should not need to know which
stage produced the gap to find the answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pricing import ENGINE_VERSION
from pricing.implied import DEFAULT_STEPS, implied_volatility

from .store import Store

#: Distinct from SOLVED / NO_SOLUTION / BRACKET_EXHAUSTED / NOT_IDENTIFIED /
#: NO_QUOTE — those are the solver's vocabulary. This one belongs to the
#: derived layer: the solver was never reached at all.
NO_MARKET_CONTEXT = "no_market_context"

#: Every exercise style currently needed. See the module docstring.
DEFAULT_STYLE = "american"


def _mid(bid: float | None, ask: float | None) -> float | None:
    """The same two-sided test ``QuoteRecord.two_sided`` applies, against raw SQL columns."""
    if bid and bid > 0 and ask and ask > 0:
        return (bid + ask) / 2
    return None


def _invert_one(row, *, style: str, steps: int) -> tuple[str, float | None, str]:
    if row["risk_free_rate"] is None or row["dividend_yield"] is None:
        return (
            NO_MARKET_CONTEXT,
            None,
            "risk-free rate or dividend yield unavailable for this snapshot",
        )
    result = implied_volatility(
        row["option_type"],
        style,
        _mid(row["bid"], row["ask"]),
        row["underlying_price"],
        row["strike"],
        row["risk_free_rate"],
        row["dividend_yield"],
        row["time_to_expiry"],
        steps=steps,
    )
    return result.status, result.implied_vol, result.detail


def derive_batch(
    store: Store,
    *,
    engine_version: int = ENGINE_VERSION,
    style: str = DEFAULT_STYLE,
    steps: int = DEFAULT_STEPS,
) -> int:
    """Invert every raw quote not yet derived at ``engine_version``.

    The incremental step: cheap after an ordinary capture, since only the
    quotes that capture just wrote are pending. Returns the number of quotes
    processed, solved or not — a contract excluded for a stated reason still
    counts, because it was still handled rather than skipped.
    """
    pending = store.pending_for_derivation(engine_version)
    computed_at = datetime.now(timezone.utc)
    for row in pending:
        status, implied_vol, detail = _invert_one(row, style=style, steps=steps)
        store.write_derived(
            quote_id=row["quote_id"],
            engine_version=engine_version,
            status=status,
            implied_vol=implied_vol,
            detail=detail,
            computed_at=computed_at,
        )
    return len(pending)


def rebuild(
    store: Store,
    *,
    engine_version: int = ENGINE_VERSION,
    style: str = DEFAULT_STYLE,
    steps: int = DEFAULT_STEPS,
) -> int:
    """Recompute every raw quote at ``engine_version``, then drop every other version.

    The wholesale half of KTD9. Reads only the raw archive to do it — proof
    that the layer is derived rather than captured, since the same input run
    through a different ``engine_version`` produces the table from scratch.

    Derives the new version *before* removing the old one, deliberately —
    wiping first (``store.clear_derived()``) would leave the table empty at
    every version for the whole recompute, so a crash partway through it, or
    a reader arriving mid-rebuild, would see nothing rather than the
    previous version's still-good numbers. Deriving first means the only
    state a crash before the final cleanup step can leave behind is the old
    and new versions coexisting — never neither.
    """
    processed = derive_batch(store, engine_version=engine_version, style=style, steps=steps)
    store.clear_stale_derived(keep_engine_version=engine_version)
    return processed


def needs_rebuild(store: Store, *, engine_version: int = ENGINE_VERSION) -> bool:
    """Whether the derived table holds rows at a version other than the current one.

    Distinguishes two situations ``derive_batch`` alone cannot tell apart:
    an archive that has simply never been derived yet (``derived_engine_versions``
    is empty — the ordinary incremental path is correct and cheap), and one
    derived at a version the engine no longer runs (a stale version is
    present — the very next incremental call would otherwise silently
    inherit the whole accumulated backlog as "pending", inside whatever
    unattended job happens to call it next, rather than as a deliberate,
    named rebuild).
    """
    versions = store.derived_engine_versions()
    return bool(versions) and versions != {engine_version}
