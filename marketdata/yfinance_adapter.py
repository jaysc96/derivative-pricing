"""The live provider, behind the neutral interface.

Everything Yahoo-shaped stops here: DataFrames become tuples of records, field
names become attributes, and library exceptions become the interface's own
error types. Downstream code cannot tell which provider it is talking to, which
is the whole point (KTD8) — and the reason the designated fallback is a new
file rather than a refactor.

``yfinance`` is imported lazily so the package imports without it. The test
suite exercises this adapter against recorded fixtures and never reaches the
network; only a live capture needs the dependency present.

**Every provider call runs under a socket timeout.** ``yfinance`` gives no
timeout parameter to pass through, and an unattended job has no worse failure
than a hang: a run that blocks forever records nothing at all, so the capture
looks neither productive nor unproductive and the fallback trigger never fires.
Setting the default socket timeout for the duration of the call bounds each
individual socket operation. It does not bound total wall clock — a server
dribbling one byte at a time stays under it indefinitely — so it is a floor
under the failure mode rather than a guarantee.
"""

from __future__ import annotations

import re
import socket
from contextlib import contextmanager
from datetime import date, datetime, timezone

from .adapter import (
    ChainSnapshot,
    MalformedResponse,
    NotSupported,
    ProviderError,
    ProviderUnavailable,
    QuoteRecord,
    RateLimited,
    UnderlyingBar,
    clean_count,
    clean_implied_vol,
    clean_price,
    validate_option_type,
    validate_strike,
)


#: Seconds any single socket operation may block. Chosen well above a healthy
#: chain pull and well below the gap between scheduled runs.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Matched against lowercased exception text. Anchored phrases, not loose
#: substrings: ``"rate"`` alone appears inside "generate", "separate",
#: "accurate" and "corporate", and every one of those would have been read as
#: a throttle — three attempts with backoff, then a `rate_limited` code that
#: says the provider is throttling us when it is doing nothing of the kind.
_RATE_LIMITED = re.compile(r"\brate[\s_-]?limit|\b429\b|\btoo many requests?\b")
_TIMED_OUT = re.compile(r"\btimed?[\s_-]?out\b|\btimeout\b")
_NETWORK = re.compile(r"\bconnection\b|\bnetwork\b|\bunreachable\b|\bssl\b|\bdns\b|"
                      r"\b(?:could not|failed to|cannot) resolve\b")


@contextmanager
def socket_timeout(seconds: float | None):
    """Bound each socket operation for the duration of the block.

    ``yfinance`` exposes no timeout parameter, so this is the only lever
    available without reaching past the library into its session.
    """
    if seconds is None:
        yield
        return
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def classify(exc: Exception) -> Exception:
    """Translate a library exception into one of ours.

    Matching on message text is unlovely, but ``yfinance`` raises bare
    exceptions for most failures and the distinction between "slow down" and
    "broken" is the one the capture job's backoff depends on.
    """
    text = str(exc).lower()
    if _RATE_LIMITED.search(text):
        return RateLimited("provider rate limited the request")
    if _TIMED_OUT.search(text):
        return ProviderUnavailable("provider timed out")
    if _NETWORK.search(text):
        return ProviderUnavailable("network failure reaching provider")
    return ProviderUnavailable(f"provider request failed ({type(exc).__name__})")


class YFinanceAdapter:
    """Yahoo Finance option chains and daily bars."""

    name = "yfinance"

    def __init__(self, ticker_factory=None, timeout: float | None = DEFAULT_TIMEOUT_SECONDS) -> None:
        """``ticker_factory`` exists for tests to inject a recorded fixture.

        ``timeout`` bounds each socket operation; ``None`` disables the bound,
        which is only sensible when no socket is involved.
        """
        self._ticker_factory = ticker_factory
        self._timeout = timeout

    def _ticker(self, symbol: str):
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - dependency-present path
            raise ProviderUnavailable(
                "yfinance is not installed; install the 'data' extra"
            ) from exc
        return yfinance.Ticker(symbol)

    def expiries(self, symbol: str) -> tuple[date, ...]:
        try:
            with socket_timeout(self._timeout):
                raw = self._ticker(symbol).options
        except ProviderError:
            # Already ours — a missing dependency, most often. Re-classifying
            # would relabel it as a failed request and lose the instruction.
            raise
        except Exception as exc:  # noqa: BLE001 - translating, not handling
            raise classify(exc) from exc
        if raw is None:
            raise MalformedResponse("expiry list is absent")
        try:
            return tuple(date.fromisoformat(str(value)) for value in raw)
        except ValueError as exc:
            raise MalformedResponse("expiry list holds a non-date value") from exc

    def option_chain(
        self, symbol: str, expiry: date, as_of: date | None = None
    ) -> ChainSnapshot:
        if as_of is not None:
            raise NotSupported(
                "yfinance serves only current chains; a dated chain needs the "
                "backfill adapter"
            )

        ticker = self._ticker(symbol)
        try:
            with socket_timeout(self._timeout):
                chain = ticker.option_chain(expiry.isoformat())
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise classify(exc) from exc

        captured_at = datetime.now(timezone.utc)
        quotes: list[QuoteRecord] = []
        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            quotes.extend(self._records(frame, option_type))

        return ChainSnapshot(
            provider=self.name,
            symbol=symbol,
            expiry=expiry,
            captured_at=captured_at,
            as_of=captured_at.date(),
            origin="live",
            underlying_price=self._underlying_price(ticker),
            quotes=tuple(quotes),
        )

    def _records(self, frame, option_type: str) -> list[QuoteRecord]:
        if frame is None:
            raise MalformedResponse(f"{option_type} frame is absent")
        required = {"contractSymbol", "strike", "bid", "ask"}
        missing = required - set(getattr(frame, "columns", []))
        if missing:
            # The failure mode a scraper actually has: a renamed field, which
            # would otherwise arrive as a column of nulls and look like a thin
            # chain rather than a broken one.
            raise MalformedResponse(f"{option_type} frame is missing {sorted(missing)}")

        records = []
        for row in frame.to_dict("records"):
            records.append(
                QuoteRecord(
                    contract_symbol=str(row["contractSymbol"]),
                    option_type=validate_option_type(option_type),
                    strike=validate_strike(row["strike"]),
                    bid=clean_price(row.get("bid"), field="bid"),
                    ask=clean_price(row.get("ask"), field="ask"),
                    last=clean_price(row.get("lastPrice"), field="lastPrice"),
                    volume=clean_count(row.get("volume"), field="volume"),
                    open_interest=clean_count(row.get("openInterest"), field="openInterest"),
                    provider_iv=clean_implied_vol(row.get("impliedVolatility")),
                    last_trade_at=_to_datetime(row.get("lastTradeDate")),
                )
            )
        return records

    def _underlying_price(self, ticker) -> float | None:
        """Best-effort. A chain without a spot is still worth keeping."""
        try:
            info = getattr(ticker, "fast_info", None)
            if info is not None:
                value = info.get("lastPrice") if hasattr(info, "get") else None
                if value is not None:
                    return clean_price(value, field="underlying_price")
        except Exception:  # noqa: BLE001 - optional field, never fatal
            return None
        return None

    def underlying_history(
        self, symbol: str, start: date, end: date
    ) -> tuple[UnderlyingBar, ...]:
        try:
            with socket_timeout(self._timeout):
                frame = self._ticker(symbol).history(
                    start=start.isoformat(), end=end.isoformat(), auto_adjust=False
                )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise classify(exc) from exc
        if frame is None:
            raise MalformedResponse("underlying history is absent")

        bars = []
        for stamp, row in zip(frame.index, frame.to_dict("records")):
            bar_date = stamp.date() if hasattr(stamp, "date") else date.fromisoformat(str(stamp))
            close = clean_price(row.get("Close"), field="Close")
            if close is None:
                continue
            bars.append(
                UnderlyingBar(
                    symbol=symbol,
                    bar_date=bar_date,
                    open=clean_price(row.get("Open"), field="Open") or close,
                    high=clean_price(row.get("High"), field="High") or close,
                    low=clean_price(row.get("Low"), field="Low") or close,
                    close=close,
                    volume=clean_count(row.get("Volume"), field="Volume"),
                )
            )
        return tuple(bars)


def _to_datetime(value) -> datetime | None:
    """Coerce the provider's last-trade stamp, which is a trade time not a quote time."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
