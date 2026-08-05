"""The live provider, behind the neutral interface.

Everything Yahoo-shaped stops here: DataFrames become tuples of records, field
names become attributes, and library exceptions become the interface's own
error types. Downstream code cannot tell which provider it is talking to, which
is the whole point (KTD8) — and the reason the designated fallback is a new
file rather than a refactor.

``yfinance`` is imported lazily so the package imports without it. The test
suite exercises this adapter against recorded fixtures and never reaches the
network; only a live capture needs the dependency present.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from .adapter import (
    ChainSnapshot,
    MalformedResponse,
    NotSupported,
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


def classify(exc: Exception) -> Exception:
    """Translate a library exception into one of ours.

    Matching on message text is unlovely, but ``yfinance`` raises bare
    exceptions for most failures and the distinction between "slow down" and
    "broken" is the one the capture job's backoff depends on.
    """
    text = str(exc).lower()
    if "rate" in text or "429" in text or "too many" in text:
        return RateLimited("provider rate limited the request")
    if "timeout" in text or "timed out" in text:
        return ProviderUnavailable("provider timed out")
    if any(token in text for token in ("connection", "network", "resolve", "ssl")):
        return ProviderUnavailable("network failure reaching provider")
    return ProviderUnavailable(f"provider request failed ({type(exc).__name__})")


class YFinanceAdapter:
    """Yahoo Finance option chains and daily bars."""

    name = "yfinance"

    def __init__(self, ticker_factory=None) -> None:
        """``ticker_factory`` exists for tests to inject a recorded fixture."""
        self._ticker_factory = ticker_factory

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
            raw = self._ticker(symbol).options
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
            chain = ticker.option_chain(expiry.isoformat())
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
            frame = self._ticker(symbol).history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=False
            )
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
